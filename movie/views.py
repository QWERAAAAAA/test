import csv
import time
import os.path
from math import sqrt
from django.contrib import messages
from django.db.models import Avg, Count, Max
from django.http import HttpResponse, request
from django.shortcuts import render, redirect, reverse
from .forms import RegisterForm, LoginForm, CommentForm
from django.views.generic import View, ListView, DetailView
from .models import User, Movie, Genre, Movie_rating, Movie_similarity, Movie_hot
from django.views.generic import ListView
from movie.models import Movie, User, Movie_rating
from movie.matrix_factorization import generate_recommendations, invalidate_user_cache




# DO NOT MAKE ANY CHANGES
BASE = os.path.dirname(os.path.abspath(__file__))


# 首页
class IndexView(ListView):
    model = Movie
    template_name = 'movie/index.html'
    paginate_by = 15
    context_object_name = 'movies'
    ordering = 'imdb_id'
    page_kwarg = 'p'

    def get_queryset(self):
        # 返回前1000部电影
        return Movie.objects.filter(imdb_id__lte=1000)

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super(IndexView, self).get_context_data(*kwargs)
        paginator = context.get('paginator')
        page_obj = context.get('page_obj')
        pagination_data = self.get_pagination_data(paginator, page_obj)
        context.update(pagination_data)
        # print(context)
        return context

    def get_pagination_data(self, paginator, page_obj, around_count=2):
        current_page = page_obj.number

        if current_page <= around_count + 2:
            left_pages = range(1, current_page)
            left_has_more = False
        else:
            left_pages = range(current_page - around_count, current_page)
            left_has_more = True

        if current_page >= paginator.num_pages - around_count - 1:
            right_pages = range(current_page + 1, paginator.num_pages + 1)
            right_has_more = False
        else:
            right_pages = range(current_page + 1, current_page + 1 + around_count)
            right_has_more = True
        return {
            'left_pages': left_pages,
            'right_pages': right_pages,
            'current_page': current_page,
            'left_has_more': left_has_more,
            'right_has_more': right_has_more
        }


class PopularMovieView(ListView):
    model = Movie_hot
    template_name = 'movie/hot.html'
    paginate_by = 15
    context_object_name = 'movies'
    page_kwarg = 'p'

    def get_queryset(self):
        # 初始化 计算评分人数最多的100部电影，并保存到数据库中
        # ######################
        # movies = Movie.objects.annotate(nums=Count('movie_rating__score')).order_by('-nums')[:100]
        # print(movies)
        # print(movies.values("nums"))
        # for movie in movies:
            # print(movie,movie.nums)
            # record = Movie_hot(movie=movie, rating_number=movie.nums)
            # record.save()
        # ######################

        hot_movies=Movie_hot.objects.all().values("movie_id")
        # print(hot_movies)
        # for movie in hot_movies:
            # print(movie)
            # print(movie.imdb_id,movie.rating_number)
        # Movie.objects.filter(movie_hot__rating_number=)
        # 一个bug!这里filter出来虽然是正确的100部电影，但是会按照imdb_id排序，导致正确的结果被破坏了！也就是得不到100部热门电影的正确顺序！
        # movies=Movie.objects.filter(id__in=hot_movies.values("imdb_id"))
        # 找出100部热门电影，同时按照评分人数排序
        # 因此我们必须要手动排序一次。另外也不太好用
        movies=Movie.objects.filter(id__in=hot_movies).annotate(nums=Max('movie_hot__rating_number')).order_by('-nums')
        return movies

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super(PopularMovieView, self).get_context_data(*kwargs)
        paginator = context.get('paginator')
        page_obj = context.get('page_obj')
        pagination_data = self.get_pagination_data(paginator, page_obj)
        context.update(pagination_data)
        # print(context)
        return context

    def get_pagination_data(self, paginator, page_obj, around_count=2):
        current_page = page_obj.number

        if current_page <= around_count + 2:
            left_pages = range(1, current_page)
            left_has_more = False
        else:
            left_pages = range(current_page - around_count, current_page)
            left_has_more = True

        if current_page >= paginator.num_pages - around_count - 1:
            right_pages = range(current_page + 1, paginator.num_pages + 1)
            right_has_more = False
        else:
            right_pages = range(current_page + 1, current_page + 1 + around_count)
            right_has_more = True
        return {
            'left_pages': left_pages,
            'right_pages': right_pages,
            'current_page': current_page,
            'left_has_more': left_has_more,
            'right_has_more': right_has_more
        }

# 分类
class TagView(ListView):
    model = Movie
    template_name = 'movie/tag.html'
    paginate_by = 15
    context_object_name = 'movies'
    # ordering = 'movie_rating__score'
    page_kwarg = 'p'

    def get_queryset(self):
        if 'genre' not in self.request.GET.dict().keys():
            movies = Movie.objects.all()
            return movies[100:200]
        else:
            movies = Movie.objects.filter(genre__name=self.request.GET.dict()['genre'])
            print(movies)
            return movies[:100]

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super(TagView, self).get_context_data(*kwargs)
        if 'genre' in self.request.GET.dict().keys():
            genre = self.request.GET.dict()['genre']
            context.update({'genre': genre})
        paginator = context.get('paginator')
        page_obj = context.get('page_obj')
        pagination_data = self.get_pagination_data(paginator, page_obj)
        context.update(pagination_data)
        return context

    def get_pagination_data(self, paginator, page_obj, around_count=2):
        current_page = page_obj.number

        if current_page <= around_count + 2:
            left_pages = range(1, current_page)
            left_has_more = False
        else:
            left_pages = range(current_page - around_count, current_page)
            left_has_more = True

        if current_page >= paginator.num_pages - around_count - 1:
            right_pages = range(current_page + 1, paginator.num_pages + 1)
            right_has_more = False
        else:
            right_pages = range(current_page + 1, current_page + 1 + around_count)
            right_has_more = True
        return {
            'left_pages': left_pages,
            'right_pages': right_pages,
            'current_page': current_page,
            'left_has_more': left_has_more,
            'right_has_more': right_has_more
        }

# 搜索
class SearchView(ListView):
    model = Movie
    template_name = 'movie/search.html'
    paginate_by = 15
    context_object_name = 'movies'
    # ordering = 'movie_rating__score'
    page_kwarg = 'p'

    def get_queryset(self):
        movies = Movie.objects.filter(name__icontains=self.request.GET.dict()['keyword'])
        print(movies)
        return movies

    def get_context_data(self, *, object_list=None, **kwargs):
        # self.genre=self.request.GET.dict()['genre']
        context = super(SearchView, self).get_context_data(*kwargs)
        paginator = context.get('paginator')
        page_obj = context.get('page_obj')
        pagination_data = self.get_pagination_data(paginator, page_obj)
        context.update(pagination_data)
        context.update({'keyword': self.request.GET.dict()['keyword']})
        return context

    def get_pagination_data(self, paginator, page_obj, around_count=2):
        current_page = page_obj.number

        if current_page <= around_count + 2:
            left_pages = range(1, current_page)
            left_has_more = False
        else:
            left_pages = range(current_page - around_count, current_page)
            left_has_more = True

        if current_page >= paginator.num_pages - around_count - 1:
            right_pages = range(current_page + 1, paginator.num_pages + 1)
            right_has_more = False
        else:
            right_pages = range(current_page + 1, current_page + 1 + around_count)
            right_has_more = True
        return {
            'left_pages': left_pages,
            'right_pages': right_pages,
            'current_page': current_page,
            'left_has_more': left_has_more,
            'right_has_more': right_has_more
        }


# 注册视图
class RegisterView(View):
    def get(self, request):
        return render(request, 'Signin.html', {'show_register': False})

    def post(self, request):
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect(reverse('movie:index'))
        else:
            errors = form.get_errors()
            # 传递一个标志变量，指示显示注册表单
            return render(request, 'Signin.html', {'register_form': form, 'register_errors': errors, 'show_register': True})

# 登录视图
class LoginView(View):
    def get(self, request):
        return render(request, 'Signin.html')

    def post(self, request):
        form = LoginForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data.get('name')
            pwd = form.cleaned_data.get('password')
            user = User.objects.filter(name=name, password=pwd).first()
            if user:
                request.session['user_id'] = user.id
                return redirect(reverse('movie:index'))
            else:
                errors = ['用户名或者密码错误!']
                return render(request, 'Signin.html', {'login_form': form, 'login_errors': errors})
        else:
            errors = form.get_errors()
            return render(request, 'Signin.html', {'login_form': form, 'login_errors': errors})

def UserLogout(request):
    # 登出，立即停止会话
    request.session.set_expiry(-1)
    return redirect(reverse('movie:index'))


class MovieDetailView(DetailView):
    '''电影详情页面'''
    model = Movie
    template_name = 'movie/detail.html'
    # 上下文对象的名称
    context_object_name = 'movie'

    def get_context_data(self, **kwargs):
        # 重写获取上下文方法，增加评分参数
        context = super().get_context_data(**kwargs)
        # 判断是否登录用
        login = True
        try:
            user_id = self.request.session['user_id']
        except KeyError as e:
            login = False  # 未登录

        # 获得电影的pk
        pk = self.kwargs['pk']
        movie = Movie.objects.get(pk=pk)

        if login:
            # 已经登录，获取当前用户的历史评分数据
            user = User.objects.get(pk=user_id)

            rating = Movie_rating.objects.filter(user=user, movie=movie).first()
            # 默认值
            score = 0
            comment = ''
            if rating:
                score = rating.score
                comment = rating.comment
            context.update({'score': score, 'comment': comment})

        similarity_movies = movie.get_similarity()
        # 获取与当前电影最相似的电影
        context.update({'similarity_movies': similarity_movies})
        # 判断是否登录，没有登录则不显示评分页面
        context.update({'login': login})

        return context

    # 接受评分表单,pk是当前电影的数据库主键id
    def post(self, request, pk):
        url = request.get_full_path()
        form = CommentForm(request.POST)
        if form.is_valid():
            # 获取分数和评论
            score = form.cleaned_data.get('score')
            comment = form.cleaned_data.get('comment')
            
            # 获取用户和电影
            user_id = request.session['user_id']
            user = User.objects.get(pk=user_id)
            movie = Movie.objects.get(pk=pk)

            # 更新一条记录
            rating = Movie_rating.objects.filter(user=user, movie=movie).first()
            if rating:
                # 如果存在则更新
                rating.score = score
                rating.comment = comment
                rating.save()
            else:
                # 如果不存在则添加
                rating = Movie_rating(user=user, movie=movie, score=score, comment=comment)
                rating.save()
            
            # 使该用户的推荐缓存失效
            invalidate_user_cache(user_id)
            
            messages.info(request, "评论成功!")
        else:
            # 表单没有验证通过
            messages.info(request, "评分不能为空!")
        return redirect(reverse('movie:detail', args=(pk,)))


class RatingHistoryView(DetailView):
    '''用户详情页面'''
    model = User
    template_name = 'movie/history.html'
    # 上下文对象的名称
    context_object_name = 'user'

    def get_context_data(self, **kwargs):
        # 这里要增加的对象：当前用户过的电影历史
        context = super().get_context_data(**kwargs)
        user_id = self.request.session['user_id']
        user = User.objects.get(pk=user_id)
        # 获取ratings即可
        ratings = Movie_rating.objects.filter(user=user)

        context.update({'ratings': ratings})
        return context


def delete_recode(request, pk):
    movie = Movie.objects.get(pk=pk)
    user_id = request.session['user_id']
    user = User.objects.get(pk=user_id)
    rating = Movie_rating.objects.get(user=user, movie=movie)
    rating.delete()
    
    # 使该用户的推荐缓存失效
    invalidate_user_cache(user_id)
    
    messages.info(request, f"删除 {movie.name} 评分记录成功！")
    # 跳转回评分历史
    return redirect(reverse('movie:history', args=(user_id,)))

# 电影推荐
class RecommendMovieView(ListView):
    model = Movie
    template_name = 'movie/recommend.html'
    paginate_by = 15
    context_object_name = 'movies'
    page_kwarg = 'p'

    def get_queryset(self):
        start_time = time.time()
        try:
            user_id = self.request.session['user_id']
            user = User.objects.get(pk=user_id)
            
            # 使用矩阵分解生成推荐
            print(f"\n===== 开始为用户 {user.name}(ID:{user.id}) 生成推荐列表 =====")
            recommended_movies, ratings_count = generate_recommendations(user, n_recommendations=30)
            
            # 计算并打印推荐耗时
            elapsed_time = time.time() - start_time
            print(f"推荐列表生成完成，耗时: {elapsed_time:.4f} 秒")
            
            # 如果评分不足，存储提示信息
            if ratings_count < 10:
                self.rating_count_insufficient = True
                self.ratings_count = ratings_count
                # 返回热门电影作为替代
                return Movie.objects.annotate(
                    rating_count=Count('movie_rating')
                ).order_by('-rating_count')[:15]
            else:
                self.rating_count_insufficient = False
            
            # 打印推荐电影信息
            print(f"推荐电影列表（共 {len(recommended_movies)} 部）:")
            for i, movie in enumerate(recommended_movies[:10], 1):
                # 获取该电影的平均评分
                avg_score = Movie_rating.objects.filter(movie=movie).aggregate(Avg('score'))['score__avg']
                if avg_score is None:
                    avg_score = "暂无评分"
                else:
                    avg_score = f"{avg_score:.1f}"
                
                # 获取该电影的评分人数
                rating_count = Movie_rating.objects.filter(movie=movie).count()
                
                # 获取电影类型
                genres = ", ".join([genre.name for genre in movie.genre.all()[:3]])
                
                print(f"{i}. {movie.name} - 平均评分: {avg_score} ({rating_count}人评价) - 类型: {genres}")
            
            if len(recommended_movies) > 10:
                print(f"... 以及其他 {len(recommended_movies) - 10} 部电影")
            print("===============================================\n")
            
            return recommended_movies
        except (KeyError, User.DoesNotExist):
            # 用户未登录或不存在，返回热门电影
            print("\n===== 用户未登录，返回热门电影列表 =====")
            self.rating_count_insufficient = False
            hot_movies = Movie.objects.annotate(
                rating_count=Count('movie_rating')
            ).order_by('-rating_count')[:15]
            
            elapsed_time = time.time() - start_time
            print(f"热门电影列表生成完成，耗时: {elapsed_time:.4f} 秒")
            print("===============================================\n")
            
            return hot_movies

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super(RecommendMovieView, self).get_context_data(*kwargs)
        paginator = context.get('paginator')
        page_obj = context.get('page_obj')
        pagination_data = self.get_pagination_data(paginator, page_obj)
        context.update(pagination_data)
        
        # 添加评分不足的提示信息到上下文
        if hasattr(self, 'rating_count_insufficient') and self.rating_count_insufficient:
            context['rating_count_insufficient'] = True
            context['ratings_count'] = self.ratings_count
            context['ratings_needed'] = 10 - self.ratings_count
        else:
            context['rating_count_insufficient'] = False
            
        return context

    def get_pagination_data(self, paginator, page_obj, around_count=2):
        current_page = page_obj.number

        if current_page <= around_count + 2:
            left_pages = range(1, current_page)
            left_has_more = False
        else:
            left_pages = range(current_page - around_count, current_page)
            left_has_more = True

        if current_page >= paginator.num_pages - around_count - 1:
            right_pages = range(current_page + 1, paginator.num_pages + 1)
            right_has_more = False
        else:
            right_pages = range(current_page + 1, current_page + 1 + around_count)
            right_has_more = True
        return {
            'left_pages': left_pages,
            'right_pages': right_pages,
            'current_page': current_page,
            'left_has_more': left_has_more,
            'right_has_more': right_has_more
        }
