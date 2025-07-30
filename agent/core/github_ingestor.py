# core/github_ingestor.py

import requests
from urllib.parse import urlparse
from core.vector_store import VectorStoreBuilder
import os

class GitHubIngestor:
    def __init__(self, github_token=None):
        self.headers = {
            "Authorization": f"token {github_token}"
        } if github_token else {}

    def _get_repo_details(self, repo_url):
        parsed = urlparse(repo_url)
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 2:
            raise ValueError("Invalid GitHub repo URL")
        return parts[0], parts[1]  # owner, repo

    def _get_default_branch(self, owner, repo):
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        response = requests.get(api_url, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"Failed to get default branch: {response.status_code} {response.text}")
        return response.json()["default_branch"]

    def _get_repo_tree(self, owner, repo, branch):
        tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        response = requests.get(tree_url, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"Failed to get repo tree: {response.status_code} {response.text}")
        data = response.json()
        if "tree" not in data:
            raise Exception(f"No 'tree' key found in response: {data}")
        return data["tree"]


    def _fetch_file(self, owner, repo, branch, file_path):
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
        print(f"Fetching file: {file_path} from {url}")
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            print(f"[Warning] Failed to fetch {file_path}: {response.status_code} {response.text}")
            return None
        print(response.text[:100])  # Debug: show part of file content
        return response.text


    def _group_files_by_component(self, tree):
        components = {}
        for item in tree:
            if item["type"] != "blob":
                continue
            path = item["path"]
            if any(path.endswith(ext) for ext in [".py", ".js", ".ts", ".java", ".cpp", ".cs"]):
                component = path.split("/")[0]
                components.setdefault(component, []).append(path)
        return components

    def ingest_and_store(self, repo_url, index_path="vector_store/index.faiss", metadata_path="vector_store/metadata.json"):
        owner, repo = self._get_repo_details(repo_url)
        branch = self._get_default_branch(owner, repo)
        tree = self._get_repo_tree(owner, repo, branch)

        components = self._group_files_by_component(tree)
        component_sources = {}

        # Change: build component_sources as {component: {file_path: code_text, ...}, ...}
        for component, files in components.items():
            print(f"Component: {component}, Files: {len(files)}")
            if not files:
                print(f"Component {component} had no usable files")
            file_dict = {}
            for file in files:
                content = self._fetch_file(owner, repo, branch, file)
                if content is None:
                    print(f"Failed to fetch content for {file}")
                    continue
                if content:
                    file_dict[file] = content
            if file_dict:
                component_sources[component] = file_dict

        print(f"[Ingestor] Extracted {len(component_sources)} components. Saving to FAISS...")
        print("Components received:", len(component_sources))
        print("Components:", list(component_sources.keys()))
        vs = VectorStoreBuilder(index_path=index_path, metadata_path=metadata_path)
        vs.build_store(component_sources)

        print(f"[Ingestor] Successfully indexed repo: {repo_url}")
        return list(component_sources.keys())  # return list of component names
