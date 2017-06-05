from django.contrib import admin

from .models import User, Piclist, Photo


admin.site.register(User)
admin.site.register(Piclist)
admin.site.register(Photo)
