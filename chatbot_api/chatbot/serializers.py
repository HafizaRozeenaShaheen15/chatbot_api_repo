from rest_framework import serializers
from .models import ChatbotQuery

class ChatbotQuerySerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatbotQuery
        fields = '__all__'
