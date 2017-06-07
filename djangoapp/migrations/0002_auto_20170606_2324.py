# -*- coding: utf-8 -*-
# Generated by Django 1.10 on 2017-06-06 20:24
from __future__ import unicode_literals

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('djangoapp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Gallery',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=200)),
                ('task', models.CharField(blank=True, max_length=200)),
                ('date_created', models.DateField(auto_now_add=True)),
                ('date_modified', models.DateField(auto_now=True)),
                ('owner', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='gallerys', to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.RemoveField(
            model_name='piclist',
            name='owner',
        ),
        migrations.RemoveField(
            model_name='photo',
            name='piclist',
        ),
        migrations.DeleteModel(
            name='Piclist',
        ),
        migrations.AddField(
            model_name='photo',
            name='gallery',
            field=models.ForeignKey(default=0, on_delete=django.db.models.deletion.CASCADE, related_name='photos', to='djangoapp.Gallery'),
            preserve_default=False,
        ),
    ]
