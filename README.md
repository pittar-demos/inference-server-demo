# Red Hat AI Inference Server (vLLM) Demo

In this demo, we will run a [Granite model](https://huggingface.co/RedHatAI/granite-3.1-8b-instruct-quantized.w4a16) using Red Hat AI Inference Server (vLLM) on OpenShift.

## Prerequisites

* Node Feature Discovery (NFD) operator deployed and a default instance created.
* Nvidia GPU operator deployed and a default instance created to install drivers.
* OpenShift Pipelines operator deployed (to enable the pipline that downloads models from Huggingface).

## Getting Started:  Download a Model

First, we will create a new project called `inference-demo` as well as a simple Tekton pipeline to download a model from Huggingface into a PVC.  To do this, run:

```
oc apply -k manifests/pipeline
```

From the OpenShift UI, navigate to the `inference-demo` project and find the new Pipelines. Start the pipeline and use the following as the model to download, and make sure to select the PVC named `models` as the model workspace.

```
RedHatAI/granite-3.1-8b-instruct-quantized.w4a16
```

This pipeline will take a few minutes, as it is downloading the Granite LLM from Huggingface.  When it's done (green) you can deploy Red Hat AI Inference Server. 

## Run Your Model

 This next step will create a standard `Deployment` using the Nvidia version of the Red Hat supported vLLM image (Red Hat AI Inference Server) that serves the model that you just downloaded to the `models` PVC.  It also creates a `Service` and a `Route` so the model can be accessed.

```
oc apply -k manifests/inferenceserver
```

Depending on your hardware, it will take a few minutes to fully initiallize.  You can follow the pod logs to watch the progress.  The model will be ready when you see `INFO: Application startup complete.` in the logs. Once this is loaded up, you can get the URL for your inference server and send your first request!

```
INFERENCE_URL=$(oc get route granite -o jsonpath='{.spec.host}' -n inference-demo)
```

```
curl -X POST https://$INFERENCE_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite-3.1-8b-instruct-quantized.w4a16",
    "messages": [{"role": "user", "content": "What is AI?"}],
    "temperature": 0.1
  }'
```

## Adding A Chat UI

Now that your model is up and running, let's add a chat UI!  For this, we will deploy an instance of [AnythingLLM]().

In the `/manifests/anythingllm` directory, update `config-secret.yaml` by changing the value of `GENERIC_OPEN_AI_BASE_PATH` to the path of your LLM (make sure to include `/v1` at the end).

With that done, apply this directory:

```
oc apply -k manifests/anythingllm
```

This should spin up pretty fast.  Once it's ready, hit the new route and start chatting with your model!

## Monitoring

For bonus marks, you can add some LLM and GPU monitoring to your cluster!

### NVidia GPU Monitoring

NVidia provides the [configuration](https://docs.nvidia.com/datacenter/cloud-native/openshift/latest/enable-gpu-monitoring-dashboard.html) to add a GPU dashboard to the native OpenShift "Observe" dashboards.

The required `ConfigMap` already exists in this repository.  To add it to your cluster, run:

```
oc apply -f manifests/monitoring/nvidia/nvidia-dashboard-configmap.yaml
```

Now, when you go to "Observe -> Dashboards" you should find a new "NVIDIA DCGM Exporter Dashboard" available that shows stats for power usage, temperature, gpu utilization, etc...

### vLLM Stats Dashaboards

This step will requires that you have already installed the Red Hat "Cluster Observability Operator" (COO) and the community Grafana operator.

The next step will create a new namespace called `llm-monitoring` where Grafana and Prometheus will be deployed.  It will also create the required `GrafanaDataSource` and `GrafanaDashboard` instances and a `ServiceMonitor` to scrape metrics from the vLLM deployment.

```
oc apply -k manifests/monitoring
```