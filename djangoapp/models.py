from django.db import models
from django.db.models.signals import post_save
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from django.dispatch import receiver

class Piclist(models.Model):
    owner = models.ForeignKey(
        'auth.User',
        related_name='piclists',
        on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    task = models.CharField(max_length=200, blank=True)
    date_created = models.DateField(auto_now_add=True)
    date_modified = models.DateField(auto_now=True)
    def __str__(self):
        return "{}".format(self.name)

class Photo(models.Model):
    owner = models.ForeignKey(
        'auth.User',
        related_name='photos',
        on_delete=models.CASCADE)
    image = models.ImageField(upload_to="%Y/%m/%d")#URLField(max_length=500)
    task = models.CharField(max_length=200, blank=True)
    piclist = models.ForeignKey(Piclist, related_name='photos', on_delete=models.CASCADE)

    def __str__(self):
        return "{}".format(self.name)

@receiver(post_save, sender=User)
def create_auth_token(sender, instance=None, created=False, **kwargs):
    if created:
        Token.objects.create(user=instance)
