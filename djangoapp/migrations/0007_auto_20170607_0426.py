# -*- coding: utf-8 -*-
# Generated by Django 1.10 on 2017-06-07 01:26
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('djangoapp', '0006_remove_gallery_slug'),
    ]

    operations = [
        migrations.AlterField(
            model_name='photo',
            name='image',
            field=models.ImageField(height_field='height', upload_to='', width_field='width'),
        ),
    ]