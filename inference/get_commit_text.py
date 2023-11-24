import requests as re
import sys
import json


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

# Example of script execution:
# python3 get_commit_text.py https://github.com/google/jax/commit/5a5730d9fcc3b59f8cdc590f3a774d22fa250162
# or
# use function run("https://github.com/google/jax/commit/5a5730d9fcc3b59f8cdc590f3a774d22fa250162")
if __name__ == "__main__":
    assert (
        len(sys.argv) == 2
    ), "\nInput format: \npython3 commit_info_from_url.py $commit link from github$"

    link = sys.argv[1]

    responce = run(link)
    json_responce = json.loads(responce)
    commit_info_printing(json_responce)
