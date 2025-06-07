import sys, os
from loguru import logger

# 获得当前项目的绝对路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(root_dir, "logs")  # 存放项目日志目录的绝对路径

if not os.path.exists(log_dir):  # 如果日志目录不存在，则创建
    os.mkdir(log_dir)

# LOG_FILE = "translation.log"  # 存储日志的文件

# Trace < Debug < Info < Success < Warning < Error < Critical

class MyLogger:
    def __init__(self):
        # log_file_path = os.path.join(log_dir, LOG_FILE)
        self.logger = logger  # 写日志的对象
        # 清空所有设置
        self.logger.remove()
        # 添加控制台输出的格式,sys.stdout为输出到屏幕;关于这些配置还需要自定义请移步官网查看相关参数说明
        self.logger.add(sys.stdout, level='DEBUG',
                        format="<green>{time:YYYYMMDD HH:mm:ss}</green> | "  
                               "{process.name} | "  
                               "{thread.name} | "  
                               "<cyan>{module}</cyan>.<cyan>{function}</cyan>"  
                               ":<cyan>{line}</cyan> | "  
                               "<level>{level}</level>: "  
                               "<level>{message}</level>",
                        )

    def get_logger(self):
        return self.logger


log = MyLogger().get_logger()

if __name__ == '__main__':
    print('str.pdf'['str.pdf'.rindex('.'):])
    # @log.catch  # 整个函数自动加上try， catch。自动捕获异常，并且通过日志打印
    def test():
        try:
            print(3/0)
        except ZeroDivisionError as e:
            # log.error(e)
            log.exception(e)  # 以后常用
    test()
