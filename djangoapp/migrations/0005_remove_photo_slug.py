# -*- coding: utf-8 -*-
# Generated by Django 1.10 on 2017-06-06 21:15
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('djangoapp', '0004_auto_20170607_0012'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='photo',
            name='slug',
        ),
    ]
