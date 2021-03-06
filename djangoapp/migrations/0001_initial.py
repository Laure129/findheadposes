# -*- coding: utf-8 -*-
# Generated by Django 1.10 on 2017-06-06 19:32
from __future__ import unicode_literals

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Photo',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(height_field='height', upload_to='%Y/%m/%d', width_field='width')),
                ('width', models.IntegerField(default=0)),
                ('height', models.IntegerField(default=0)),
                ('pitch', models.FloatField(default=-180)),
                ('yaw', models.FloatField(default=-180)),
                ('task', models.CharField(blank=True, max_length=200)),
                ('date_uploaded', models.DateField(auto_now_add=True)),
            ],
            options={
                'ordering': ['-yaw'],
            },
        ),
        migrations.CreateModel(
            name='Piclist',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=200)),
                ('task', models.CharField(blank=True, max_length=200)),
                ('date_created', models.DateField(auto_now_add=True)),
                ('date_modified', models.DateField(auto_now=True)),
                ('owner', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='piclists', to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.AddField(
            model_name='photo',
            name='piclist',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='photos', to='djangoapp.Piclist'),
        ),
    ]
