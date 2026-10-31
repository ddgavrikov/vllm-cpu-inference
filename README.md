# CPU inference with vLLM in Kubernetes

Тестовое задание Data Engineer. Автор: Гавриков Дмитрий.

OpenAI-compatible API для `Qwen/Qwen2.5-0.5B-Instruct`, vLLM **0.27.0+cpu**, только CPU. Среда: Linux x86_64, CPython 3.12, полноценный Kubernetes либо Minikube. Выбранная instruct-модель открыта, не требует HF-токена и помещается в 32 ГБ RAM локального компьютера. Для предсказуемой работы на AVX2 используется float32 без квантизации: около 2 ГБ на веса, 1 ГБ KV cache плюс служебная память. Под ограничен 8 GiB. GPU-ресурсы не запрашиваются; Docker build проверяет CPU-сборку PyTorch и CPU-платформу vLLM.

## Состав решения

| Требование | Реализация |
|---|---|
| Собственный Dockerfile и непривилегированный пользователь | `Dockerfile`, UID/GID 10001, `USER`, restricted pod security |
| Все Python-зависимости зафиксированы | `requirements.txt`, включая транзитивные пакеты, точные версии и SHA256; CPU wheel привязан к commit |
| Приватные логи и факт обращения | `app/privacy.py`, `app/launch.py`, `logging.json` |
| PVC и загрузка только при первом запуске | `scripts/download_model.py`, `k8s/pvc.yaml` |
| HF и API токены через Secret | `scripts/create_secret.py`, `secretKeyRef` в deployment |
| Параметры без пересборки образа | `k8s/configmap.yaml`, `server.json` |
| Бонус 1 kustomize | `kubectl apply -k k8s` |
| Бонус 2 отдельный rootless initContainer | `Dockerfile.init`, отдельный небольшой образ без PyTorch/vLLM |
| Инструмент для бонуса 3 | `scripts/benchmark.py`, streaming TTFT/latency/throughput |

## Локальный запуск

Нужны Docker с Linux containers, kubectl, Python 3 и Minikube. Рекомендуется выделить Minikube 6 CPU, 12 ГБ RAM и 25 ГБ диска. Команды запускаются из корня репозитория в Bash; команды Docker, kubectl и Python подходят также PowerShell. Docker Desktop Kubernetes можно использовать вместо Minikube: пропустите создание кластера и `minikube image load`.

```bash
minikube start --driver=docker --cpus=6 --memory=12288 --disk-size=25g
docker build --platform linux/amd64 -t llm-cpu:0.1.0 .
docker build --platform linux/amd64 -f Dockerfile.init -t llm-downloader:0.1.0 .
minikube image load llm-cpu:0.1.0
minikube image load llm-downloader:0.1.0

python scripts/create_secret.py
kubectl apply -k k8s
kubectl -n llm-cpu rollout status deployment/llm --timeout=900s
kubectl -n llm-cpu port-forward svc/llm 8000:8000
```

Скрипт Secret запрашивает HF-токен и API-токен скрытым вводом. Для публичной модели HF-токен оставьте пустым; для gated-модели укажите read-токен с разрешённым доступом к выбранной модели. API-токен можно задать самостоятельно либо сгенерировать пустым вводом. Токен не печатается и не попадает в файл или аргументы процесса. Объект передаётся kubectl через stdin. Для дальнейшего использования автоматически сгенерированный токен можно прочитать скрытым способом в переменную среды:

```bash
export API_TOKEN="$(kubectl -n llm-cpu get secret llm-secrets -o jsonpath='{.data.api-token}' | base64 --decode)"
python scripts/benchmark.py --requests 8 --concurrency 2
```

PowerShell:

```powershell
$encodedApiToken = kubectl -n llm-cpu get secret llm-secrets -o 'jsonpath={.data.api-token}'
$env:API_TOKEN = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($encodedApiToken))
python scripts/benchmark.py --requests 8 --concurrency 2
```

В другом терминале можно проверить обычный запрос без вывода токена в командной строке:

```bash
python scripts/client.py
```

Без корректного `Authorization: Bearer <token>` запросы `/v1/...` отклоняются с 401. `/health` используется probes без токена. Сервис имеет тип ClusterIP; доступ с рабочей машины — через port-forward. Для внешней публикации следует добавить TLS на ingress. Kubernetes Secret хранит base64, а не шифрует данные: в промышленном кластере нужны RBAC и encryption at rest. HF-токен доступен только initContainer; основному контейнеру он не передаётся.

## Кэш и конфигурация

В ConfigMap зафиксирована точная HF-ревизия `7ae557604adf67be50417f59c2c2f167def9a775`. initContainer скачивает safetensors и файлы токенизатора в `/models/<revision>.partial`, проверяет наличие основных файлов, пишет манифест размеров и атомарно переименовывает каталог. Незавершённая загрузка не считается готовой. Linux `flock` защищает от двух писателей. Токен не сохраняется в PVC.

При готовом кэше сначала проверяются локальные файлы, затем скрипт завершается с `model_cache_hit`, **не импортируя HF-клиент и не выполняя сетевых вызовов**. Изменение размера/отсутствие файла вызывает повторную загрузку. Манифест размеров выявляет обрыв/удаление, но не является криптографической проверкой порчи того же размера. Основной контейнер монтирует PVC read-only, получает абсолютный локальный путь, `HF_HUB_OFFLINE=1` и `TRANSFORMERS_OFFLINE=1`. Образ должен быть предварительно доступен на узле; отсутствие повторного обращения к HF не отменяет возможный pull самого образа Kubernetes.

Параметры `dtype`, длина контекста, batch budget, число последовательностей, prefix caching и chunked prefill меняются в `k8s/configmap.yaml`. Переменные CPU KV cache и OpenMP также берутся из ConfigMap. Секреты и параметры приватности переопределить через `server.json` нельзя.

```bash
kubectl apply -k k8s
kubectl -n llm-cpu rollout restart deployment/llm
kubectl -n llm-cpu rollout status deployment/llm --timeout=900s
kubectl -n llm-cpu logs deployment/llm -c download-model
```

Для другой модели замените MODEL_ID и **40-символьный commit** MODEL_REVISION, проверьте поддержку архитектуры CPU backend и бюджет RAM/PVC. Для gated-моделей сначала предоставьте доступ HF-токену. PVC переживает пересоздание pod; удаление namespace удаляет PVC и может удалить веса согласно reclaim policy. Один replica и стратегия Recreate исключают гонки attach для ReadWriteOnce при обновлении. Это локальный стенд, не HA-инсталляция.

## Приватное логгирование

Штатные `enable-log-requests` и `enable-log-outputs` выключены. Собственный logging formatter не читает текст `LogRecord`, аргументы, traceback и stack info: он выдаёт только фиксированный тип события и уровень. ASGI middleware отдельно фиксирует завершение обращения: случайный серверный UUID, HTTP-статус, длительность. Он не читает тело, заголовки, query string, клиентский request ID или текст ответа и не буферизует streaming.

Супервизор перехватывает оба потока дочернего vLLM-процесса и пропускает только заново собранные события с разрешёнными числовыми/enum/UUID полями. Неструктурированные `print`, исключения и сторонние сообщения отбрасываются. Пример строки:

```json
{"event":"inference_request","request_id":"471531ad-1321-4dfd-aa22-cdcd808ac218","status":200,"elapsed_ms":1214.5}
```

Цена строгой приватности — отсутствие подробных сообщений ошибок vLLM в stdout/stderr; для диагностики остаются exit code, probes, Kubernetes events и уровни ошибок. Не включайте debug-дампы и трассировку содержимого на реальных пользовательских данных. Проверка `scripts/smoke.py` отправляет контрольную строку в корректном и ошибочном запросах и проверяет её отсутствие вместе с токеном в логах.

## Проверки

```bash
python -m unittest discover -s tests -v
kubectl kustomize k8s
python scripts/smoke.py
```

Unit-тесты проверяют приватность formatter/supervisor, streaming middleware и кэш с подменённым загрузчиком; они не заменяют реальный инференс. На Windows только `flock` в unit-тесте заменён заглушкой; в Linux используется настоящий lock. Workflow `.github/workflows/verify.yaml` собирает оба образа, поднимает полноценный кластер kind, проверяет UID, CPU backend, авторизацию и логи, пересоздаёт pod с тем же PVC и запускает benchmark. Артефакты содержат CPU, версию Kubernetes, JSON измерений и таблицу.

## Нагрузочное тестирование

```bash
python scripts/benchmark.py --requests 8 --concurrency 2 --output results/benchmark.json
```

Один прогрев исключается из статистики для каждого сценария. Короткий и длинный промпты фиксированы в скрипте; фактические входные токены и число выходных токенов берутся из server usage. Генерация ограничена 32 токенами с `ignore_eos=true`. TTFT — время от начала HTTP-запроса до первой SSE-дельты с непустым текстом, latency — до завершения потока. Throughput считается по wall time всей серии, в req/s и output tokens/s. Есть p50/p95; на 8 запросах p95 — максимум выборки, для устойчивых выводов увеличьте число повторов. Ошибки не исключаются из статистики молча: тест прекращается.

Измерено 31 октября 2026 года: GitHub Actions `ubuntu-24.04`, CPU `AMD EPYC 9V74 80-Core Processor`, 4 vCPU, Kubernetes `v1.34.0` в kind; контейнер: 2 OpenMP-потока, float32, KV cache 1 GiB, prefix caching выключен, max model len 2048. По 8 запросов на сценарий, concurrency 2, 32 выходных токена на запрос. Эти показатели относятся к проверочному runner, а не к локальному Windows-компьютеру.

| Prompt | Input tokens | TTFT p50/p95 s | Latency p50/p95 s | req/s | output tok/s |
|---|---:|---:|---:|---:|---:|
| short | 35 | 0.344/0.418 | 3.388/3.514 | 0.589 | 18.864 |
| long | 1001 | 5.715/7.799 | 10.551/11.406 | 0.186 | 5.943 |

[Успешный полный прогон](https://github.com/ddgavrikov/vllm-cpu-inference/actions/runs/41194141326). Сырые выборки: `results/benchmark.json`; оборудование и версия кластера: `results/cpu.txt`, `results/kubernetes.json`. Прогон подтвердил запуск в Kubernetes, UID 10001, CPU backend, отказ без Bearer token, отсутствие контрольного текста и токена в логах, повторный запуск с `model_cache_hit`. Все 8 unit-тестов прошли. Машиночитаемый протокол: `results/validation.json`.

### Как повышать throughput

При одинаковых 32 выходных токенах увеличение входа с 35 до 1001 токена повысило медиану TTFT с 0,344 до 5,715 с и снизило выходной throughput с 18,864 до 5,943 токена/с. Это согласуется с ростом стоимости prefill и ожидания в очереди; для разделения этих затрат нужно отдельное профилирование. Настройки ниже — направления экспериментов, а не уже измеренные улучшения.

**Много коротких запросов.** Постепенно увеличивать `max-num-seqs` и `max-num-batched-tokens`, чтобы больше запросов делили один шаг исполнения. Контролировать p95 TTFT и объём KV cache, не выбирать batch больше доступной RAM. Подбирать `OMP_NUM_THREADS` и привязку потоков под выделенные физические ядра; оставить ресурсы API-процессу. При одинаковом системном префиксе включить prefix caching и отдельно измерить эффект на тёплом кэше. Репликация возможна после выделения отдельных PVC/правильного RWX-хранилища и ресурсов; простое увеличение replicas не подходит текущему локальному RWO-стенду.

**Мало запросов с большим контекстом.** Увеличить `max-model-len` только до нужной длины, предварительно выделив KV cache через `VLLM_CPU_KVCACHE_SPACE` и memory limit. Сохранить chunked prefill; подбирать `max-num-batched-tokens`, чтобы крупный prefill не блокировал decoding других последовательностей. Уменьшать `max-num-seqs`, если контексты вытесняют KV cache и приводят к preemption. При повторяющемся длинном префиксе оценить prefix caching. Для поддерживающего BF16 CPU отдельно сравнить `dtype=bfloat16` с float32; на AVX2 экономия памяти не гарантирует ускорения. Настройки для GPU вроде `gpu-memory-utilization` не решают ограничений CPU. Каждое изменение проверять тем же benchmark на одинаковом оборудовании и длинах запросов.

## Воспроизводимость зависимостей

Установка выполняется с `--require-hashes`; pip не должен подбирать новые версии. Все 153 Python-пакета основного образа, включая pip и транзитивные зависимости, зафиксированы; в облегчённом образе — 14 пакетов. Python base image закреплён release tag и SHA256 digest. Системные библиотеки устанавливаются из Debian bookworm: полная побитовая воспроизводимость ОС требует также snapshot apt-репозитория. Lock ориентирован только на CPython 3.12/Linux x86_64, не предназначен для нативной установки vLLM в Windows.

Обновление lock-файлов выполняется отдельно от сборки (uv 0.9.11):

```bash
uv pip compile requirements.in --python-version 3.12 --python-platform x86_64-manylinux_2_34 --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --generate-hashes -o requirements.txt
uv pip compile requirements-init.in --python-version 3.12 --python-platform x86_64-manylinux_2_34 --generate-hashes -o requirements-init.txt
```

## Источники

- [vLLM 0.27.0 CPU Dockerfile](https://github.com/vllm-project/vllm/blob/v0.27.0/docker/Dockerfile.cpu)
- [Аргументы API server 0.27.0](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/entrypoints/openai/cli_args.py)
- [Документация CPU backend](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/)
- [Модель и лицензия Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Использование Kubernetes Secrets](https://kubernetes.io/docs/concepts/configuration/secret/)
