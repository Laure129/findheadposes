#!/usr/bin/env python
from django.db import models
from django.contrib.auth.models import User

class Gallery(models.Model):
    owner = models.ForeignKey(
        'auth.User',
        related_name='galleries',
        on_delete=models.CASCADE)
    title = models.CharField(max_length=200) #для лучшего результата введите имя на нескольких языкай используйте популярные сокращения через запятую
    task = models.CharField(max_length=200, blank=True)
    date_created = models.DateField(auto_now_add=True)
    date_modified = models.DateField(auto_now=True)

    class Meta:
        ordering = ['date_modified']

    def __str__(self):
        return self.title

class Photo(models.Model):
    image = models.ImageField(null = False, blank=False, width_field='width', height_field='height')#
    width = models.IntegerField(default=0)
    height = models.IntegerField(default=0)
    pitch = models.FloatField(default=-180)
    yaw = models.FloatField(default=-180)
    task = models.CharField(max_length=200, blank=True)
    gallery = models.ForeignKey(Gallery, related_name='photos', on_delete=models.CASCADE)
    date_uploaded = models.DateField(auto_now_add=True)

    class Meta:
        ordering = ['-yaw']
