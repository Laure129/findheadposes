# -*- coding: utf-8 -*-
# Generated by Django 1.10 on 2017-06-06 20:31
from __future__ import unicode_literals

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('djangoapp', '0002_auto_20170606_2324'),
    ]

    operations = [
        migrations.AlterField(
            model_name='gallery',
            name='owner',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='galleries', to=settings.AUTH_USER_MODEL),
        ),
    ]
