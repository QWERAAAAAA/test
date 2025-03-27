from django.core.management.base import BaseCommand
from movie.matrix_factorization import run_matrix_factorization

class Command(BaseCommand):
    help = '运行矩阵分解算法并保存结果到数据库'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('开始执行矩阵分解...'))
        run_matrix_factorization()
        self.stdout.write(self.style.SUCCESS('矩阵分解完成!'))