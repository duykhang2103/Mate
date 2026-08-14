import modal
import subprocess
import os

app = modal.App("prunevid")

# 1. Define the environment
# Add any other pip packages your pipeline requires (like vllm, accelerate, etc.)
app_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .apt_install("python3-dev", "build-essential", "wget", "git", "ffmpeg", "unzip")
    .pip_install("huggingface_hub", "torch", "torchvision") # Thêm ffmpeg để xử lý video
    .workdir("/workspace")
    .add_local_file("requirements.clean.txt", remote_path="/workspace/requirements.clean.txt", copy=True)
    
    # 2. Chạy cài đặt thư viện. 
    # BƯỚC NÀY SẼ ĐƯỢC CACHE CHẶT CHẼ. Nó chỉ chạy lại nếu nội dung file requirements.txt thay đổi.
    .run_commands("pip install -r requirements.clean.txt")
    .run_commands("pip install -U huggingface_hub")
    .add_local_dir(".", remote_path="/workspace", copy=True)
)

# 2. Create a Volume to permanently store your final submission.json
output_vol = modal.Volume.from_name("prunevid", create_if_missing=True)
dataset_vol = modal.Volume.from_name("prunevid-dataset", create_if_missing=True)
# 3. Configure the cloud function
@app.function(
    image=app_image,
    gpu="L4",           # Attaches a powerful GPU (can also be "L4" or "A100")
    cpu=4.0,
    timeout=86400,        # 24-hour maximum timeout
    volumes={
        "/outputs": output_vol,
        "/dataset": dataset_vol   # <--- Dataset sẽ nằm ở đây
    },
    # This automatically uploads your current local folder to the cloud container
    # mounts=[modal.Mount.from_local_dir(".", remote_path="/workspace")] 
)
def run_pipeline():
    import os
    import sys
    import json
    import subprocess
    os.chdir("/workspace")

    # ==========================================
    # 1. TẠO SYMLINK ĐỂ TRỎ VÀO VOLUME
    # ==========================================
    # Tạo các thư mục vật lý nằm an toàn bên trong Volume
    # subprocess.run("mkdir -p /dataset/DATAS/MVBench /dataset/MODELS /outputs/results", shell=True)
    
    # # Tạo liên kết ảo: Code của bạn lưu vào ./DATAS nhưng thực chất nó ghi thẳng vào /dataset/DATAS
    # subprocess.run("ln -snf /dataset/DATAS/MVBench ./DATAS/MVBench", shell=True)
    # subprocess.run("ln -snf /dataset/MODELS ./MODELS", shell=True)
    # subprocess.run("ln -snf /dataset/results ./results", shell=True)

    subprocess.run("mkdir -p /dataset/DATAS/Video-MME /dataset/MODELS /dataset/Video-MME/results", shell=True, check=True)
    
    # 2. CREATE LOCAL PARENT FOLDERS FIRST
    subprocess.run("mkdir -p ./DATAS", shell=True, check=True)
    
    # 3. Create the symlinks
    subprocess.run("ln -snf /dataset/DATAS/Video-MME ./DATAS/Video-MME", shell=True, check=True)
    subprocess.run("ln -snf /dataset/MODELS ./MODELS", shell=True, check=True)
    
    # FIX: Point this to /outputs/results, not /dataset/results
    subprocess.run("ln -snf /dataset/Video-MME/results ./results", shell=True, check=True)

    import torch
    # ... (Giữ nguyên phần in thông số GPU của bạn) ...[cite: 4]
    
    cmd = """ 
    hf auth login --token SECRET && bash videomme-eval-prunevid.sh
    """
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()

    # ==========================================
    # 2. LƯU VĨNH VIỄN DỮ LIỆU
    # ==========================================
    output_vol.commit()   # Lưu kết quả infer từ /outputs[cite: 4]
    dataset_vol.commit()  # <--- THÊM DÒNG NÀY: Lưu Model và Video tải về từ /dataset
    print("✅ Pipeline finished! Toàn bộ Model, Dataset và Results đã được lưu an toàn.")



# def run_pipeline():
#     # Navigate to the uploaded code
#     import os
#     import sys  # <--- Add this line here
#     import json
#     import subprocess
#     os.chdir("/workspace")

#     import torch
#     print("=== GPU & SYSTEM VERIFICATION ===")
#     print(f'CUDA available: {torch.cuda.is_available()}')
#     if torch.cuda.is_available():
#         print(f'GPU: {torch.cuda.get_device_name(0)}')
#         print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
#         print(f'Free VRAM: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB')

#     print("=================================\n")
    
#     cmd = """ 
#     nvidia-smi && hf auth login --token SECRET && bash mvbench-eval.sh
    
#     """
    
#     # Execute the bash command and stream logs to the dashboard
#     process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
#     for line in process.stdout:
#         print(line, end="")
#     process.wait()

#     # Save the volume state so the file is stored permanently
#     output_vol.commit()
#     # print("✅ Pipeline finished! submission.json safely stored in Volume.")