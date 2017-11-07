# -*- coding: utf-8 -*-
from PIL import Image
# from json import dumps
import numpy as np
from . import recognize_img
from django.core.urlresolvers import reverse
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.shortcuts import render
from myproject.myapp.forms import DocumentForm
from myproject.myapp.models import Document


def list(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()

            # Redirect to the document list after POST
            return HttpResponseRedirect(reverse('list'))
    else:
        form = DocumentForm()  # A empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()

    my_document = None
    for document in documents:
        my_document = document

    # Render list page with the documents and the form
    return render(
        request,
        'list.html',
        {'documents': my_document, 'form': form}
    )


graph_type = {
    0: "line chart",
    1: "bar chart",
    2: "pie chart"
}


def classify(request):
    img_url = request.GET.get("img_url")
    img_url = img_url[1:]
    """
    img_url 是你上传的图片的路径，下面除了 return 语句之外的语句是你对这个图片做的操作，这是需要重写的部分。
    """
    img = Image.open(img_url)
    image = img.convert('L').resize((56, 56))
    batch_x = np.fromstring(image.tobytes(), dtype=np.uint8).reshape(1, 56 * 56)
    batch_x = batch_x * (1. / 255) - 0.5
    prediction_y = recognize_img.classifier(batch_x)
    return HttpResponse(graph_type[prediction_y[0]], content_type="application/json")
    # return HttpResponse(graph_type[0], content_type="application/json")
