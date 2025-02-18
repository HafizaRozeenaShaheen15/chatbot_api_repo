from django.urls import path
from .views import chatbot_api

# urlpatterns = [
#     path('chat/', chatbot_api, name="chatbot_api"),
# ]

urlpatterns = [
    path('v1/chat/', chatbot_api, name="chatbot_api"),
]
