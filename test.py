from main import TransformerLanguageModel
import tiktoken
import torch

text = "you create a sense of familiarity, comfort, and shared interests, making it easier to communicate and influence their decision-making process. \
Your level of enthusiasm can be contagious and can greatly impact the customer's perception of you and your offering. By displaying genuine enthusiasm and confidence in what you are selling, you inspire trust and make the customer more inclined to believe in the value of your product or service.\
Creating a positive first impression is not just about superficial charm or manipulation; it is about genuinely caring for the customer and their needs. It is about establishing a strong foundation of trust and credibility, setting the stage for a successful sales journey. By mastering the art of creating a positive first impression, you lay the groundwork for building lasting relationships and achieving sales success. Chapter 1: Building Rapport and Capturing Attention \
Subpoint: Active Listening and Demonstrating Empathy"

model = TransformerLanguageModel()
# 加载模型权重
model.load_state_dict(torch.load('model-ckpt.pt'))
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 设备
model = model.to(device=device)

if __name__ == '__main__':
    model.eval()
    max_token_number = int(input('请输入需要续写的文本长度:'))
    encoding = tiktoken.get_encoding('cl100k_base')
    tokenized_text = encoding.encode(text)
    tokenized_text = torch.tensor(tokenized_text,dtype = torch.long, device = device)
    tokenized_text = tokenized_text.unsqueeze(0)
    tokenized_text = tokenized_text.to(device)
    print(tokenized_text)
    text = model.generate(tokenized_text, max_token_number)
    print('续写结果:')
    y = encoding.decode(text[0].tolist())
    print(y)
