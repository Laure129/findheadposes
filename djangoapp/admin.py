from django.contrib import admin
from .models import User, Gallery, Photo

class PhotoAdmin(admin.ModelAdmin):
    list_display = ['image', 'task', 'gallery']

    class Meta:
        model = Photo

admin.site.register(Gallery)
admin.site.register(Photo, PhotoAdmin)
