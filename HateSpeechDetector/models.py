from django.db import models

# Create your models here.


class Comment(models.Model):
    content = models.TextField(max_length=1000000)
    isHate = models.BooleanField()