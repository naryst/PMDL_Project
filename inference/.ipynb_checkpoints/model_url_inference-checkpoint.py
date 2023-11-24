import requests as re
import sys
import json
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, pipeline
from transformers import AutoModelForSeq2SeqLM
import torch


# Transforms link from the form of:
# https://github.com/$USERNAME$/$REPO NAME$/commit/$COMMIT SHA$
# To:
# https://api.github.com/repos/$USERNAME$/$REPO NAME$/commits/$COMMIT SHA$
def transform_link_to_api_form(link):
    split_link = link.split("/")
    split_link[2] = "api." + split_link[2]
    split_link = split_link[:3] + ["repos"] + split_link[3:]
    split_link[6] = split_link[6] + "s"
    api_link = "/".join(split_link)
    return api_link


def commit_info_answer(commit_json, link):
    repo_name = link.split("/")[4]
    commit_message = commit_json["commit"]["message"]
    files_changes = commit_json["files"]

    value = {
        "repository_name": repo_name,
        "commit_message": commit_message,
        "files": [],
    }
    for index, file in enumerate(files_changes):
        append_dict = {
            "filename": file["filename"],
            "status": file["status"],
            "additions": file["additions"],
            "deletions": file["deletions"],
            "total_changes": file["changes"],
            # "code_changes": file["patch"],
        }

        try:
            if append_dict["total_changes"] != 0:
                append_dict["code_changes"] = file["patch"]
        except KeyError:
            append_dict["code_changes"] = "Commit changes is too big, not supported"

        value["files"].append(append_dict)
    return json.dumps(value)


def commit_info_printing(commit_json):
    print()
    print("Repository name: " + commit_json["repository_name"])
    print("Commit message: " + commit_json["commit_message"])
    print("=" * 70)

    files_changes = commit_json["files"]
    for file in files_changes:
        print("Filename: " + file["filename"])
        print("Status: " + file["status"])
        print("Additions: " + str(file["additions"]))
        print("Deletions: " + str(file["deletions"]))
        print("Total changes: " + str(file["total_changes"]))
        if "code_changes" in file:
            print("Code changes:")
            print(file["code_changes"])
        print("=" * 70)


def run(link: str):
    api_link = transform_link_to_api_form(link)
    print(api_link)
    response = re.get(api_link)
    assert response.ok, "Invalid commit link"

    commit_json = response.json()

    return commit_info_answer(commit_json, link)


def model_inference(model, tokenizer, text, return_dict=False, seqs=5):
    prompt = text
    input = tokenizer(
        prompt, return_tensors="pt", truncation=True, padding="max_length"
    ).to(device)
    model.eval()
    with torch.no_grad():
        sample_outputs = model.generate(
            **input,
            max_new_tokens=25,
            top_k=50,
            num_return_sequences=seqs,
            num_beams=5,
            no_repeat_ngram_size=2,
            do_sample=True,
            early_stopping=True,
            top_p=0.95,
        )
    if not return_dict:
        for i, sample_output in enumerate(sample_outputs):
            print(
                "{}: {}".format(
                    i, tokenizer.decode(sample_output, skip_special_tokens=True)
                )
            )
            print("-" * 80)
    else:
        res = []
        for i, sample_output in enumerate(sample_outputs):
            res.append(tokenizer.decode(sample_output, skip_special_tokens=True))
        return res


# Example of script execution:
# python3 commit_info_from_url.py https://github.com/google/jax/commit/5a5730d9fcc3b59f8cdc590f3a774d22fa250162
# or
# use function run("https://github.com/google/jax/commit/5a5730d9fcc3b59f8cdc590f3a774d22fa250162")
if __name__ == "__main__":
    assert (
        len(sys.argv) == 2
    ), "\nInput format: \npython3 commit_info_from_url.py $commit link from github$"

    link = sys.argv[1]

    responce = run(link)
    json_responce = json.loads(responce)
    # commit_info_printing(json.loads(response))
    model_input = ""
    # print(json_responce['files'])
    for file in json_responce["files"]:
        model_input += file["filename"] + "\n"
        model_input += file["code_changes"] + "\n\n\n"

    #DIR_PREFIX = "/home/user/commits/commit_messages_generation/"
    #checkpoint = DIR_PREFIX + "model/t5p_CommitChron_v2/checkpoint-225000"
    checkpoint = "narySt/codeT5p_CommitMessageGen"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(
        device
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    print("GENERATED COMMIT MESSAGES:")
    model_inference(model, tokenizer, model_input)
    print("ORIGINAL MESSAGE:")
    print(json_responce["commit_message"])
