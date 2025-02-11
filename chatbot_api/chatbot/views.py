from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from rest_framework.decorators import api_view
from .models import ChatbotQuery
from .serializers import ChatbotQuerySerializer
from .chatbot_logic import chatbot_response

@api_view(['POST'])
def chatbot_api(request):
    user_input = request.data.get("query", "").strip()
    
    if not user_input:
        return JsonResponse({"error": "No input provided"}, status=400)

    response = chatbot_response(user_input)

    # Save query in database
    chatbot_query = ChatbotQuery.objects.create(user_input=user_input, response=response)
    serializer = ChatbotQuerySerializer(chatbot_query)

    return JsonResponse(serializer.data)
