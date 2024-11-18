# Gemini 1.5 Pro in Local Python



Full tutorial is [here](https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/quickstart-multimodal#local-shell)



## Set up gcloud

1. Install

```cmd
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh # you need to choose adding this path into .bashrc
gcloud init
```

2. Login

```cmd
gcloud auth application-default login
```

3. create project: it might be optional since it will automatically ask you to do this during login

```cmd
gcloud projects create <your-project-ID>
```

The <your-project-ID> should be a universal unqiue ID



## Install Package

```bash
pip install google-generativeai
pip install vertexai
```





## Enable API

You can find your project on https://console.cloud.google.com/vertex-ai?hl=en&project=axial-engine-429103-n1

Search API in the search bar



## Sample Code

```python

import vertexai
from vertexai.generative_models import GenerativeModel, Part
import time
import google.generativeai as genai

# TODO(developer): Update the GOOGLE_API_KEY and the project_id 
GOOGLE_API_KEY='AIzaSyCAUOH-xtSh7TspZWdYtOu0QWkO6oYjJnw'
genai.configure(api_key=GOOGLE_API_KEY)
project_id = "axial-engine-429103-n1"
vertexai.init(project=project_id, location="us-central1")

video_file_name = "/home/leik/yichen/RobotLLM/DATAS/TRAIN_TEST/videos/robot_activity_videos/CLVR-CLVR-success-2023-05-09-Tue_May__9_01:31:33_2023_16787047.mp4"
print(f"Uploading file...")
video_file = genai.upload_file(path=video_file_name)
while video_file.state.name == "PROCESSING":
    print('.', end='')
    time.sleep(10)
    video_file = genai.get_file(video_file.name)
if video_file.state.name == "FAILED":
    raise ValueError(video_file.state.name)
print(f"Retrieved file '{video_file.display_name}' as: {video_file.uri}")

prompt = "Describe this video."

# Make the LLM request.
# Set the model to Gemini 1.5 Pro.
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

# Make the LLM request.
print("Making LLM inference request...")
response = model.generate_content([prompt, video_file],
                                  request_options={"timeout": 600})
# The official way to print results but does not work in the terminal
# print(response.text)

import pdb; pdb.set_trace()
# Alternative solution
# print(response.candidates[0].content.parts[0]._raw_part.text)
print(response.candidates[0].content.parts[0].text)

# delete
genai.delete_file(video_file.name)
print(f'Deleted file {video_file.uri}')
```

