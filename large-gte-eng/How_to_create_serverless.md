Как собирать serverless.

Гайд будет по этой инструкции https://docs.runpod.io/cli/projects/get-started 

Для начала нужно скачать себе runpodctl, сделать это можно, например, так (MacOS):

brew install runpod/runpodctl/runpodctl

Далее генерим себе API_KEY на ранподе в настройках (RW права) - https://www.runpod.io/console/user/settings

После создаем себе локальное рабочее окружение и вставляем туда наш ключ (лучше сохранять ключи в .env, но можно и захардкодить):

runpodctl config --apiKey $(RUNPOD_API_KEY)

После стартуем проект:

runpodctl project create

Здесь нам предложат на выбор готовые шаблоны, я выбирал Hello World. Далее нужно выбрать версию CUDA и Python. 



Идем в src/handler.py и там делаем всю нужную нам логику. В целом, можно делать много функций, импортировать их из других областей и т.д. Главное, чтобы у вас была одна функция handler(название неважно, но лучше оставить такое), которая запускает пайплайн. 

В event будет приходить ваш запрос, можно проверить работоспособность запустив:

python3 src/handler.py --test_input '{"input": {"sentences": "kek"}}'

Если все вывелось правильно, можно создавать дев окружение:

runpodctl project dev

Тут надо будет подождать, пока pod полностью запустится (можно отслеживать в UI). Если после запуска понимаем, что не докинули важные либы в requirements.txt, то можно это сделать в одноименном файле в папке builder, среда сама автоматически обновится, не надо перезагружать под.

После, можно отправлять CURL запрос в дев окружение:

curl -X 'POST' \
  'https://${YOUR_ENDPOINT}-8080.proxy.runpod.net/runsync' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": {"sentences": "kek"}
}'

Если все ок, то сервер отдаст вам ответ (в нашем случае это словарь с ключом “embeddings”).

На этом моменте можно выйти из сервака через CTRL+C, и задеплоить в serverless:

runpodctl project deploy

Это займет немного времени, в конце выдаст что-то вроде:

The following URLs are available:
    - https://api.runpod.ai/v2/${YOUR_ENDPOINT}/runsync
    - https://api.runpod.ai/v2/${YOUR_ENDPOINT}/run
    - https://api.runpod.ai/v2/${YOUR_ENDPOINT}/health

Все, теперь можно стучаться в серверлесс (после первого запроса он еще будет инициализироваться, поэтому первый запрос может выполниться не быстро): 

curl -X 'POST' \
  'https://api.runpod.ai/v2/${YOUR_ENDPOINT}/runsync' \
  -H 'accept: application/json' \
  -H  'authorization: ${YOUR_API_KEY}' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": {"sentences": "kek"}
}'

Или вместо CURL можно отправлять запросы в UI, для этого надо в runpod зайти во вкладку serverless, найти поднятый нами эндпоинт, зайти в него и нажать requests:



