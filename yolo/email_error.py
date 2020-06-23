from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
import time

import smtplib


def _format_addr(s):
    name, addr = parseaddr(s)
    return formataddr((Header(name, 'utf-8').encode(), addr))


def send_error_report(string):
    # Email地址和口令
    from_addr = 'chuyitongxue@sina.com'
    password = '58bb176086e63a8c'
    # 收件人地址
    to_addr = '903265986@qq.com'
    # SMTP服务器地址
    smtp_server = 'smtp.sina.com'
    # 参数1.邮件正文  参数2MIME的subtype,plain表示纯文本，最终的MIME就是text/plain，
    # 最后一定要用utf-8编码保证多语言的兼容性
    # 如果发送HTML邮件 把HTML字符串传进去 把plain变为html
    msg = MIMEText(string, 'plain', 'utf-8')
    # 注意不能简单地传入name <addr@example.com>，因为如果包含中文，需要通过Header对象进行编码。
    msg['From'] = _format_addr('chuyitongxue <%s>' % from_addr)
    # msg['To']接收的是字符串而不是list，如果有多个邮件地址，用,分隔即可。
    # 你看到的收件人的名字很可能不是我们传入的管理员，因为很多邮件服务商在显示邮件时，
    # 会把收件人名字自动替换为用户注册的名字，但是其他收件人名字的显示不受影响。
    msg['To'] = _format_addr('松是一个字 <%s>' % to_addr)
    t = time.strftime("%Y-%m-%d %X", time.localtime())
    msg['Subject'] = Header(t + ' 错误报告', 'utf-8').encode()

    # SMTP协议默认端口是25
    server = smtplib.SMTP(smtp_server, 25)
    # 打印出和SMTP服务器交互的所有信息。
    # SMTP协议就是简单的文本命令和响应。
    # server.set_debuglevel(1)
    # 用来登录SMTP服务器
    server.login(from_addr, password)
    # 发邮件，可以发给多个人，所以是一个list，邮件正文是一个str，as_string()把MIMEText对象变成str
    server.sendmail(from_addr, [to_addr], msg.as_string())
    server.quit()
