
from django.db import models

class ChatbotQuery(models.Model):
    user_input = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user_input
