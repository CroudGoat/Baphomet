from huggingface_hub import HfApi
api=HfApi()
rid="CloudGoat/Baphomet-1.1-4B"
api.create_repo(rid, private=True, exist_ok=True, repo_type="model")
api.upload_folder(repo_id=rid, folder_path="Baphomet-1.1-4B", repo_type="model", commit_message="Upload Baphomet-1.1-4B")
info=api.model_info(rid, files_metadata=True)
print("private",info.private,[ (s.rfilename,s.size) for s in info.siblings])
