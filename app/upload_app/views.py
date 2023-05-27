import math
from django.shortcuts import render
from django.http import JsonResponse
from PIL import Image
from django.views.decorators.csrf import csrf_exempt
import base64, io, os


@csrf_exempt
def home(request):
    template_name = 'index.html'

    if request.method == "GET":
        return render(request, template_name)
    
    if request.method == "POST":
        uploaded_file = request.FILES['image']  

        encoded_file = base64.b64encode(uploaded_file.read()).decode('utf-8')

        filename, file_extension = os.path.splitext(uploaded_file.name)
        output_format = file_extension[1:].upper() 
    
        image = Image.open(uploaded_file)

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Set quality
        quality = 70
        if quality < 0:
            quality = 0
        elif quality > 100:
            quality = 100

        # Compress image
        compressed_image = io.BytesIO()
        
        image.save(compressed_image, format=output_format, quality=quality, optimize=True)
        
        compressed_file = base64.b64encode(compressed_image.getvalue()).decode('utf-8')
        
        # Calculate compression ratio
        original_size = len(encoded_file)
        compressed_size = len(compressed_file)
        compression_ratio = math.ceil((compressed_size / original_size))

        print(compression_ratio)
    
        return JsonResponse({"image_data": encoded_file, "compress_data": compressed_file})
        


