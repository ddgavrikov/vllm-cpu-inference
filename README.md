# vLLM на CPU в Kubernetes

Тестовое задание Data Engineer — Гавриков Дмитрий.

Сервис запускает `Qwen/Qwen2.5-0.5B-Instruct` через vLLM `0.27.0+cpu` и принимает запросы в формате OpenAI API. Используется CPU, без квантизации, с `dtype=float32`. Небольшая модель выбрана для локального запуска: около 2 ГБ памяти занимают веса, ещё 1 ГБ выделен под KV cache. Лимит памяти контейнера — 8 GiB.

## Запуск

Нужны Docker с Linux containers, Minikube, kubectl и Python 3. Образы рассчитаны на Linux x86_64 с AVX2 или AVX512. Для Minikube рекомендуется выделить 6 CPU, 12 ГБ RAM и 25 ГБ диска.

Из корня репозитория:

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

При использовании Kubernetes из Docker Desktop создание Minikube и загрузку образов в него нужно пропустить.

`create_secret.py` запрашивает токены скрытым вводом и создаёт Kubernetes Secret через stdin, без записи значений в файл или аргументы процесса. Для публичной Qwen HF-токен можно оставить пустым. Для gated-модели нужен read-токен с доступом к ней. Пустой API-токен будет сгенерирован автоматически.

В другом терминале получите API-токен в переменную и отправьте запрос:

```bash
export API_TOKEN="$(kubectl -n llm-cpu get secret llm-secrets -o jsonpath='{.data.api-token}' | base64 --decode)"
python scripts/client.py
```

В PowerShell:

```powershell
$encodedApiToken = kubectl -n llm-cpu get secret llm-secrets -o 'jsonpath={.data.api-token}'
$env:API_TOKEN = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($encodedApiToken))
python scripts/client.py
```

Без правильного Bearer token запросы `/v1/...` получают 401. `/health` открыт для проверок Kubernetes. Сервис доступен внутри кластера и через port-forward; при внешней публикации потребуется TLS. Для рабочего кластера также нужны RBAC и шифрование Secrets at rest: base64 в Secret само по себе не является шифрованием.

## Как это устроено

- `Dockerfile` собирает основной образ с CPU wheel vLLM. `requirements.txt` фиксирует все Python-зависимости, включая транзитивные, и их SHA256. При сборке выполняются `pip check` и проверка CPU backend.
- `Dockerfile.init` собирает отдельный небольшой образ загрузчика без PyTorch и vLLM. Оба контейнера запускаются с UID/GID 10001, без повышения привилегий, с read-only корневой файловой системой.
- `scripts/download_model.py` скачивает веса и токенизатор в PVC. HF-ревизия закреплена полным commit hash. Загрузка идёт во временный каталог; после её завершения каталог переименовывается. `flock` не даёт двум загрузчикам писать одновременно.
- При перезапуске загрузчик проверяет локальный манифест файлов и возвращает `model_cache_hit` без обращения к HF. Проверяются наличие и размеры файлов; изменение содержимого без изменения размера эта проверка не обнаруживает.
- Основной контейнер читает веса из PVC, смонтированного read-only. `HF_HUB_OFFLINE=1` и `TRANSFORMERS_OFFLINE=1` запрещают ему обращаться к HF. HF-токен передаётся только загрузчику, API-токен — только серверу.
- `k8s/configmap.yaml` содержит параметры инференса и CPU. Манифесты применяются через kustomize. Используется одна реплика и стратегия `Recreate`, чтобы при обновлении не было конфликта подключения RWO-диска.

Параметры модели меняются без пересборки образа:

```bash
kubectl apply -k k8s
kubectl -n llm-cpu rollout restart deployment/llm
kubectl -n llm-cpu rollout status deployment/llm --timeout=900s
kubectl -n llm-cpu logs deployment/llm -c download-model
```

Для другой модели нужно изменить `MODEL_ID`, `MODEL_REVISION` и проверить бюджет памяти и поддержку её архитектуры CPU backend. PVC сохраняется при пересоздании pod; удаление namespace может удалить и веса.

## Логи

Штатное логирование запросов и генераций выключено. `app/privacy.py` записывает только UUID обращения, HTTP-статус и длительность; текст запроса, ответа и исключений не попадает в запись. Обработчик не буферизует streaming-ответ.

`app/launch.py` запускает vLLM отдельным процессом и фильтрует оба его потока вывода. В stdout проходят только события с разрешёнными полями. Это закрывает и обход через обычный `print` или traceback. Обратная сторона — подробные сообщения ошибок тоже отбрасываются; для диагностики остаются уровни ошибок, exit code, probes и Kubernetes events.

## Проверки и нагрузочный тест

```bash
python -m unittest discover -s tests -v
kubectl kustomize k8s
python scripts/smoke.py
python scripts/benchmark.py --requests 8 --concurrency 2
```

Последние две команды требуют работающий port-forward и `API_TOKEN`. Unit-тесты проверяют обработку логов, кэш, незавершённую загрузку и SSE-поток. На Windows системный `flock` в unit-тестах заменён заглушкой; в Linux используется настоящий lock. `smoke.py` проверяет авторизацию, генерацию, ошибочный запрос и отсутствие контрольной строки и токена в логах. Workflow `.github/workflows/verify.yaml` собирает образы, поднимает Kubernetes через kind, проверяет перезапуск с PVC и выполняет нагрузочный тест.

Перед каждым сценарием выполняется один прогрев. В таблице — по 8 запросов, concurrency 2 и по 32 выходных токена. TTFT считается до первой SSE-дельты с текстом, latency — до завершения потока. Throughput — число завершённых запросов или выходных токенов за время всей серии. Число токенов берётся из server usage. На такой небольшой выборке p95 совпадает с максимумом, поэтому для устойчивого сравнения стоит увеличить число запросов.

Замер от 31 октября 2026 года: GitHub Actions `ubuntu-24.04`, AMD EPYC 9V74 80-Core Processor, 4 vCPU, Kubernetes `v1.34.0`, 2 OpenMP-потока, float32, KV cache 1 GiB, контекст 2048, prefix caching выключен.

| Промпт | Входные токены | TTFT p50/p95, с | Latency p50/p95, с | Запросов/с | Выходных токенов/с |
|---|---:|---:|---:|---:|---:|
| Короткий | 35 | 0.347/0.425 | 3.479/3.508 | 0.579 | 18.536 |
| Длинный | 1001 | 5.754/7.794 | 10.919/11.393 | 0.183 | 5.856 |

[Полный прогон](https://github.com/ddgavrikov/vllm-cpu-inference/actions/runs/41199701331). Прошли 13 unit-тестов, проверка API, приватности логов и перезапуска с кэшем. Исходные измерения и сведения о машине находятся в `results/`.

### Что менять для повышения throughput

**Много коротких запросов.** Начать с `max-num-seqs` и `max-num-batched-tokens`: больший batch позволяет обрабатывать больше запросов за один шаг. При этом нужно следить за p95 TTFT и расходом KV cache. Число OpenMP-потоков подбирать под доступные физические ядра, оставив ресурсы API-процессу. При повторяющемся системном префиксе можно проверить prefix caching.

**Мало запросов с длинным контекстом.** Сначала оценить память под KV cache и нужный `max-model-len`, затем подбирать бюджет `max-num-batched-tokens` с включённым chunked prefill. Слишком большой prefill может задерживать decoding других запросов. Если возникают вытеснения KV cache, уменьшить `max-num-seqs` или увеличить выделенную память. BF16 имеет смысл отдельно сравнить на поддерживающем его CPU. Это направления для экспериментов; их эффект в приведённом тесте не измерялся.

## Обновление зависимостей

Lock-файлы рассчитаны на CPython 3.12 и Linux x86_64. Базовый Python-образ закреплён по digest. Системные пакеты берутся из Debian bookworm; для побитового повторения сборки ОС понадобился бы также snapshot apt-репозитория.

```bash
uv pip compile requirements.in --python-version 3.12 --python-platform x86_64-manylinux_2_34 --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --generate-hashes -o requirements.txt
uv pip compile requirements-init.in --python-version 3.12 --python-platform x86_64-manylinux_2_34 --generate-hashes -o requirements-init.txt
```

Использовался uv 0.9.11. Сборка образов устанавливает только wheels с проверкой хешей.

## Документация

- [CPU backend vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/)
- [CPU Dockerfile vLLM 0.27.0](https://github.com/vllm-project/vllm/blob/v0.27.0/docker/Dockerfile.cpu)
- [Параметры API server 0.27.0](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/entrypoints/openai/cli_args.py)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
