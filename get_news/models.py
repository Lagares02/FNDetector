from django.db import models

class News(models.Model):
    title = models.CharField(max_length=50)
    description = models.TextField()
    url = models.URLField(max_length=300)
    public_date = models.DateTimeField(auto_now=False, auto_now_add=False)
    image = models.URLField(max_length=300, null=True, blank=True)
    status = models.BooleanField(default=False)
    verificado = models.BooleanField(default=False)

    def __str__(self):
        return self.title

    def have_image(self):
        return self.image != ""


    
