# Red Hat AI Inference Server (vLLM) Demo

In this demo, we will run a [Granite model](https://huggingface.co/RedHatAI/granite-3.1-8b-instruct-quantized.w4a16) using Red Hat AI Inference Server (vLLM) on OpenShift.

First, create a new namespace for this demo called `inference-demo`.

```
oc new-project inference-demo
```

Next, create a Tekton pipeline to download the model from Huggingface to a local `PersistentVolumeClaim`.

```
oc apply -k manifests/pipeline
```

Using the Pipline GUI, start the pipeline and use the following as the model to download:

```
RedHatAI/granite-3.1-8b-instruct-quantized.w4a16
```

You will also need to pick your workspace storage.  Under workspace, click on the dropdown list and choose "PersistentVolumeClaim".  A new list will appear.  Select "models" from this list.

You can now start your pipeline!  This pipeline will take a few minutes, as it is downloading the Granite LLM from Huggingface.  When it's done (green) you can deploy Red Hat AI Inference Server.

```
oc apply -k manifests/inferenceserver
```

Depending on your hardware, it will take a few minutes to fully initiallize.  Once it does, you can get the URL for your inference server and send your first request!

```
INFERENCE_URL=$(oc get route granite -o jsonpath='{.spec.host}')
```

```
curl -X POST https://$INFERENCE_URL$/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite-3.1-8b-instruct-quantized.w4a16",
    "messages": [{"role": "user", "content": "What is AI?"}],
    "temperature": 0.1
  }'
```