
from django.contrib import admin
from django.urls import path
from upload_app.views import home

urlpatterns = [
    path('admin/', admin.site.urls),
    path('home/', home, name='upload_view'),
]
