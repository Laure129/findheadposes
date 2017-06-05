from rest_framework import serializers, generics
from django.contrib.auth.models import User
from .models import Photo, Piclist
from django.contrib.auth import get_user_model

UserModel = get_user_model()

class UserSerializer(serializers.ModelSerializer):

    def create(self, validated_data):
        user = UserModel.objects.create(
            username=validated_data['username'],
            email = validated_data['email'],
            first_name=validated_data['first_name'],
            last_name=validated_data['last_name'],
        #    is_active=True
        )
        user.set_password(validated_data['password'])
        user.save()
        return user

       # user = UserModel.objects.get
    class Meta:
        model = User
        fields = ('id', 'first_name', 'last_name', 'email', 'username', 'password')
        write_only_fields = ('password',)
        read_only_fields = ('id',)

class PhotoSerializer(serializers.ModelSerializer):
    image = serializers.ReadOnlyField(source='image.url')
    owner = serializers.ReadOnlyField(source='owner.username')
    class Meta:
        model = Photo
        fields = ('id', 'owner', 'image', 'task')

class PiclistSerializer(serializers.ModelSerializer):
    photos = serializers.StringRelatedField(many=True, read_only=True)
    owner = serializers.ReadOnlyField(source='owner.username')
    class Meta:
        model = Piclist
        fields = ('id', 'name', 'owner','date_created', 'date_modified', 'photos')
