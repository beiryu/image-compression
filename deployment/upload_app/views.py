import io
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from PIL import Image
import numpy as np
import base64, os, cv2, time

from upload_app.helpers.jpeg_encoder import jpeg_encoder

base_dir = os.path.dirname(__file__)

@csrf_exempt
def home(request):
    template_name = 'index.html'

    if request.method == "GET":
        return render(request, template_name)
    
    if request.method == "POST":

        quality = int(request.GET.get('quality', 80))
        
        uploaded_file = request.FILES['image']  

        encoded_file = base64.b64encode(uploaded_file.read()).decode('utf-8')

        filename, file_extension = os.path.splitext(uploaded_file.name)
        output_format = file_extension[1:].upper() 
    
        image = Image.open(uploaded_file)

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image_in_dir = os.path.join(base_dir, f'static/in/{filename}.{file_extension}')
        image_pnm_dir = os.path.join(base_dir, f'static/pnm/{filename}.pnm')
        image_out_dir = os.path.join(base_dir, f'static/out/{filename}.{output_format}')
        
        image.save(image_in_dir)
        image.save(image_pnm_dir)

        img = cv2.imread(image_pnm_dir, cv2.IMREAD_COLOR)
        height, width = img.shape[:2]

        start = time.time()
        jpeg_encoder(image_out_dir, img, height, width, quality)
        end = time.time()

        with open(image_out_dir, "rb") as image_file:
            compressed_file = base64.b64encode(image_file.read()).decode('utf-8')

        encoded_file_os = os.stat(image_in_dir)
        compressed_file_os = os.stat(image_out_dir)
        encoded_file_bytes = encoded_file_os.st_size / 1024
        compressed_file_bytes = compressed_file_os.st_size / 1024
        
        return JsonResponse({
            "image_data": encoded_file, 
            "compress_data": compressed_file,
            "image_size": round(encoded_file_bytes, 2),
            "compress_size": round(compressed_file_bytes, 2),
            "percent": round((encoded_file_bytes - compressed_file_bytes) / encoded_file_bytes * 100)
        })