"""
qq邮箱发送
"""
import smtplib
from email.mime.text import MIMEText
import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def _prepare(subject, html_content=None, text_content=None, attachments=()):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    if html_content:
        part2 = MIMEText(html_content, 'html')
        msg.attach(part2)
        # 普通文本内容
    if text_content:
        part1 = MIMEText(text_content, 'plain')
        msg.attach(part1)
    if attachments:
        for attach_file in attachments:
            part = MIMEApplication(open(attach_file, 'rb').read())
            filename = os.path.split(attach_file)[1]
            part.add_header('Content-ID', '<' + filename + '>')
            part.add_header('Content-Disposition', 'attachment', filename=filename)
            msg.attach(part)

    return msg


def _login(username, password):
    client = smtplib.SMTP_SSL('smtp.qq.com', smtplib.SMTP_SSL_PORT)
    print("连接到邮件服务器成功")

    client.login(username, password)
    return client


def _send(_from, _to, msg, client):
    msg['From'] = _from
    msg['To'] = _to
    client.sendmail(_from, _to, msg.as_string())
    print("邮件成功")


def send(subject,content):

    msg = _prepare(subject, content,attachments=('book_perspective.bmp',))
    client = _login("2065019247@qq.com", "pjxqxeuawfvlbceh")
    _send("2065019247@qq.com", "jiangzhengfool@live.com", msg, client)


if __name__ == '__main__':
    send('没有主题','附件有一个照片')

