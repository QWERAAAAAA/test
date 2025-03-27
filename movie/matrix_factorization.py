import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
import json
import time
from datetime import datetime, timedelta
from django.db import transaction
from django.db.models import Count, Max

from movie.models import (
    User, Movie, Movie_rating, 
    MatrixFactorization, UserFactors, MovieFactors,
    RecommendationCache
)


class ALSMatrixFactorization:
    """基于交替最小二乘法(ALS)的矩阵分解推荐算法"""
    
    def __init__(self, n_factors=20, n_iterations=10, regularization=0.1):
        """
        初始化ALS模型
        
        参数:
        - n_factors: 潜在因子数量(特征向量维度)
        - n_iterations: 迭代次数
        - regularization: 正则化系数
        """
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.regularization = regularization
    
    def build_user_item_matrix(self):
        """构建用户-电影评分矩阵"""
        # 获取所有用户和电影的ID映射
        users = list(User.objects.all().values_list('id', flat=True))
        movies = list(Movie.objects.all().values_list('id', flat=True))
        
        # 创建ID到索引的映射
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(users)}
        self.movie_id_map = {movie_id: idx for idx, movie_id in enumerate(movies)}
        
        # 反向映射(索引到ID)
        self.idx_to_user_id = {idx: user_id for user_id, idx in self.user_id_map.items()}
        self.idx_to_movie_id = {idx: movie_id for movie_id, idx in self.movie_id_map.items()}
        
        # 获取所有评分数据
        ratings = Movie_rating.objects.all().values_list('user_id', 'movie_id', 'score')
        
        # 构建稀疏矩阵的行、列和数值
        row_indices = []
        col_indices = []
        data = []
        
        for user_id, movie_id, rating in ratings:
            if user_id in self.user_id_map and movie_id in self.movie_id_map:
                row_indices.append(self.user_id_map[user_id])
                col_indices.append(self.movie_id_map[movie_id])
                data.append(float(rating))
        
        # 创建CSR格式的稀疏矩阵
        self.ratings_matrix = sp.csr_matrix(
            (data, (row_indices, col_indices)), 
            shape=(len(users), len(movies))
        )
        
        return self.ratings_matrix
    
    def _solve_for_user_factors(self, item_factors, ratings_matrix, lambda_val):
        """求解用户特征向量"""
        n_users, n_items = ratings_matrix.shape
        n_factors = item_factors.shape[1]
        user_factors = np.zeros((n_users, n_factors))
        
        # 对每个用户求解
        for u in range(n_users):
            # 获取该用户的评分项
            item_indices = ratings_matrix[u].indices
            if len(item_indices) == 0:
                continue
                
            # 获取对应的评分
            ratings = ratings_matrix[u].data
            
            # 构建线性系统 Ax = b
            A = item_factors[item_indices].T @ item_factors[item_indices] + lambda_val * np.eye(n_factors)
            b = item_factors[item_indices].T @ ratings
            
            # 使用共轭梯度法求解
            user_factors[u], _ = cg(A, b, tol=1e-6)
            
        return user_factors
    
    def _solve_for_item_factors(self, user_factors, ratings_matrix, lambda_val):
        """求解电影特征向量"""
        n_users, n_items = ratings_matrix.shape
        n_factors = user_factors.shape[1]
        item_factors = np.zeros((n_items, n_factors))
        
        # 对每个电影求解
        for i in range(n_items):
            # 获取评价该电影的用户
            user_indices = ratings_matrix.T[i].indices
            if len(user_indices) == 0:
                continue
                
            # 获取对应的评分
            ratings = ratings_matrix.T[i].data
            
            # 构建线性系统 Ax = b
            A = user_factors[user_indices].T @ user_factors[user_indices] + lambda_val * np.eye(n_factors)
            b = user_factors[user_indices].T @ ratings
            
            # 使用共轭梯度法求解
            item_factors[i], _ = cg(A, b, tol=1e-6)
            
        return item_factors
    
    def factorize(self):
        """执行矩阵分解"""
        print("开始矩阵分解...")
        start_time = time.time()
        
        # 构建评分矩阵
        ratings_matrix = self.build_user_item_matrix()
        n_users, n_items = ratings_matrix.shape
        
        # 随机初始化特征向量
        np.random.seed(42)  # 确保结果可重复
        user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # 交替优化用户和电影特征向量
        for iteration in range(self.n_iterations):
            print(f"迭代 {iteration+1}/{self.n_iterations}")
            
            # 固定电影特征，优化用户特征
            user_factors = self._solve_for_user_factors(item_factors, ratings_matrix, self.regularization)
            
            # 固定用户特征，优化电影特征
            item_factors = self._solve_for_item_factors(user_factors, ratings_matrix, self.regularization)
        
        self.user_factors = user_factors
        self.item_factors = item_factors
        
        print(f"矩阵分解完成，用时：{time.time() - start_time:.2f}秒")
        
        return user_factors, item_factors
    
    @transaction.atomic
    def save_to_database(self):
        """将分解结果保存到数据库"""
        print("保存矩阵分解结果到数据库...")
        
        # 创建矩阵分解模型记录
        mf_model = MatrixFactorization(
            n_factors=self.n_factors,
            iterations=self.n_iterations,
            regularization=self.regularization
        )
        mf_model.save()
        
        # 批量保存用户特征
        user_factors_objects = []
        for idx, factors in enumerate(self.user_factors):
            if idx in self.idx_to_user_id:
                user_id = self.idx_to_user_id[idx]
                user_factors_objects.append(UserFactors(
                    user_id=user_id,
                    mf_model=mf_model,
                    factors=json.dumps(factors.tolist())
                ))
        
        # 批量创建用户特征记录
        if user_factors_objects:
            UserFactors.objects.bulk_create(user_factors_objects, batch_size=1000)
        
        # 批量保存电影特征
        movie_factors_objects = []
        for idx, factors in enumerate(self.item_factors):
            if idx in self.idx_to_movie_id:
                movie_id = self.idx_to_movie_id[idx]
                movie_factors_objects.append(MovieFactors(
                    movie_id=movie_id,
                    mf_model=mf_model,
                    factors=json.dumps(factors.tolist())
                ))
        
        # 批量创建电影特征记录
        if movie_factors_objects:
            MovieFactors.objects.bulk_create(movie_factors_objects, batch_size=1000)
            
        print(f"保存完成。模型ID: {mf_model.id}")
        return mf_model
    
    @classmethod
    def load_from_database(cls, mf_model_id=None):
        """从数据库加载最新的矩阵分解模型"""
        if mf_model_id is None:
            # 如果没有指定模型ID，则加载最新的模型
            try:
                mf_model = MatrixFactorization.objects.latest('last_updated')
            except MatrixFactorization.DoesNotExist:
                return None, None, None
        else:
            try:
                mf_model = MatrixFactorization.objects.get(id=mf_model_id)
            except MatrixFactorization.DoesNotExist:
                return None, None, None
        
        # 加载用户特征和电影特征
        user_factors_dict = {uf.user_id: json.loads(uf.factors) 
                             for uf in UserFactors.objects.filter(mf_model=mf_model)}
        
        movie_factors_dict = {mf.movie_id: json.loads(mf.factors) 
                              for mf in MovieFactors.objects.filter(mf_model=mf_model)}
        
        return mf_model, user_factors_dict, movie_factors_dict


def generate_recommendations(user, n_recommendations=10, mf_model_id=None, force_recalculate=False):
    """
    为指定用户生成电影推荐
    
    参数:
    - user: User对象
    - n_recommendations: 推荐电影数量
    - mf_model_id: 指定的矩阵分解模型ID
    - force_recalculate: 是否强制重新计算推荐(忽略缓存)
    """
    # 检查用户是否有至少10部评分电影
    user_ratings_count = Movie_rating.objects.filter(user=user).count()
    if user_ratings_count < 10:
        # 如果评分不足10部，返回空列表
        return [], user_ratings_count
    
    # 检查缓存
    if not force_recalculate:
        # 检查是否有有效的缓存
        cache = RecommendationCache.objects.filter(
            user=user, 
            is_valid=True,
            created_at__gt=datetime.now() - timedelta(days=1)  # 1天内的缓存视为有效
        ).first()
        
        if cache:
            # 返回缓存的推荐结果
            recommended_movie_ids = json.loads(cache.recommended_movies)
            recommended_movies = list(Movie.objects.filter(id__in=recommended_movie_ids))
            # 按照缓存中的顺序排序
            recommended_movies.sort(key=lambda x: recommended_movie_ids.index(x.id))
            return recommended_movies, user_ratings_count
    
    # 加载模型
    mf_model, user_factors_dict, movie_factors_dict = ALSMatrixFactorization.load_from_database(mf_model_id)
    
    if not mf_model:
        # 如果没有模型，返回热门电影
        hot_movies = Movie.objects.annotate(
            rating_count=Count('movie_rating')
        ).order_by('-rating_count')[:n_recommendations]
        return list(hot_movies), user_ratings_count
    
    # 获取用户已评分的电影
    rated_movie_ids = set(Movie_rating.objects.filter(user=user).values_list('movie_id', flat=True))
    
    # 如果用户在因子矩阵中
    if user.id in user_factors_dict:
        user_vector = np.array(user_factors_dict[user.id])
        
        # 计算所有电影的预测评分
        movie_scores = []
        for movie_id, movie_vector in movie_factors_dict.items():
            if movie_id not in rated_movie_ids:  # 只推荐未评分的电影
                predicted_score = np.dot(user_vector, np.array(movie_vector))
                movie_scores.append((movie_id, predicted_score))
        
        # 增强多样性策略
        # 1. 将电影按预测分数排序
        movie_scores.sort(key=lambda x: -x[1])
        
        # 2. 从前50%的高分电影中随机选择部分电影
        top_count = min(len(movie_scores), n_recommendations * 3)
        top_candidates = movie_scores[:top_count]
        
        # 3. 使用加权随机选择，分数越高权重越大，但仍有随机性
        weights = []
        for _, score in top_candidates:
            # 使用sigmoid函数增强差异
            weight = 1.0 / (1.0 + np.exp(-score))
            weights.append(weight)
            
        # 标准化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w/total_weight for w in weights]
            
        # 随机选择不重复的电影ID
        if len(top_candidates) <= n_recommendations:
            selected_indices = range(len(top_candidates))
        else:
            # 添加随机种子，确保每次结果不同
            np.random.seed(int(time.time()))
            selected_indices = np.random.choice(
                len(top_candidates), 
                size=n_recommendations,
                replace=False,  # 不允许重复
                p=weights
            )
        
        # 获取选中的电影ID
        recommended_movie_ids = [top_candidates[i][0] for i in selected_indices]
        recommended_movies = list(Movie.objects.filter(id__in=recommended_movie_ids))
        
        # 保存到缓存
        # 首先将所有该用户的缓存标记为无效
        RecommendationCache.objects.filter(user=user).update(is_valid=False)
        # 创建新缓存
        RecommendationCache.objects.create(
            user=user,
            recommended_movies=json.dumps(recommended_movie_ids),
            is_valid=True
        )
        
        return recommended_movies, user_ratings_count
    else:
        # 用户没有评分记录或不在因子矩阵中，返回热门电影
        hot_movies = Movie.objects.annotate(
            rating_count=Count('movie_rating')
        ).order_by('-rating_count')[:n_recommendations]
        return list(hot_movies), user_ratings_count


def invalidate_user_cache(user_id):
    """当用户评分发生变化时，使该用户的推荐缓存失效"""
    # 立即失效当前用户的所有缓存
    RecommendationCache.objects.filter(user_id=user_id).update(is_valid=False)
    # 不需要提前计算新的推荐，下次访问时会自动重新计算


def run_matrix_factorization():
    """运行矩阵分解并保存结果"""
    als = ALSMatrixFactorization(n_factors=20, n_iterations=10, regularization=0.1)
    als.factorize()
    als.save_to_database()