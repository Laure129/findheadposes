#!/usr/bin/env python
from django.shortcuts import render
from .models import User, Gallery, Photo
from djangoapp.picsearch import image_bing_search, save_image
from djangoapp.dpgestimator import Estimator


def gallery_list(request):
    if request.method == "POST":
        name = request.POST.get('name')
        l1 = name.split(',')
        new_gallery = Gallery(title=l1[0], task=name, owner=request.user)
        new_gallery.save()
        l2 = []
        # отправляем запросы поочередно по списку имен
        for word in l1:
            l2.append(image_bing_search(word, 150))
        # сохраняем и загружаем в бд
        l3 = save_image(l2[0], 'media')
        for path in l3:
            try:
                # определение положения головы, загружаем в бд
                roll, pitch, yaw = Estimator('media/' + path).get_pose()
                print(roll, pitch, yaw)
                new_photo = Photo(image=path, gallery=new_gallery, pitch=pitch, yaw=yaw)
                new_photo.save()
            except:
                continue
    queryset1 = Gallery.objects.all()
    queryset2 = []
    for i in range(len(queryset1.values())):
        queryset2.append(Photo.objects.filter(gallery_id=queryset1[i].id))

    context = {
        'galleries': queryset1, 'photos_pac': queryset2
    }
    paginate_by = 20
    return render(request, 'gallery_list.html', context)

def gallery_detail(request):
    pass

def photo_detail(request):
    pass