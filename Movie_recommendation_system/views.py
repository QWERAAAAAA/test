from django.http import request
from django.shortcuts import render,redirect,reverse

def index(request):
    return render(request,'Signin.html')

def star(request):
    return render(request,'movie/star.html')