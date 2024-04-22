# LLAMA 2 Inference

This example require some extra dependencies. Here's an easy way to get it running on a fresh server.

Don't forget to put in your huggingface token from https://huggingface.co/settings/tokens

```bash
#!/bin/bash


# if you don't insert it, you will be prompted to log in later;
# you may need to rerun this script after logging in
YOUR_HF_TOKEN="insert token for headless" 

# clone and install dependencies
sudo apt install -y git
git clone https://github.com/nod-ai/SHARK-Turbine.git
cd SHARK-Turbine
pip install -r requirements.txt
pip install --update "huggingface_hub[cli]" transformers sentencepiece protobuf

# do an editable install from the cloned SHARK-Turbine
pip install --editable .

# Log in with Hugging Face CLI if token setup is required
if [[ $YOUR_HF_TOKEN == hf_* ]]; then
    huggingface login --token $YOUR_HF_TOKEN
    echo "Logged in with YOUR_HF_TOKEN."
elif [ -f ~/.cache/huggingface/token ]; then
    # Read token from the file
    TOKEN_CONTENT=$(cat ~/.cache/huggingface/token)
    
    # Check if the token starts with "hf_"
    if [[ $TOKEN_CONTENT == hf_* ]]; then
        echo "Already logged in with a Hugging Face token."
    else
        echo "Token in file does not start with 'hf_'. Please log into huggingface to download models."
        huggingface-cli login
    fi
else
    echo "Please log into huggingface to download models."
    huggingface-cli login
fi

# Step 7: Run the Python script
python examples/llama2_inference/stateless_llama.py
```
