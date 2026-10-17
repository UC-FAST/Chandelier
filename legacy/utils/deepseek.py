import time

import ollama

# 流式输出


def api_generate(text: str):
    stream = ollama.generate(
        stream=True,
        model='qwen2.5:0.5b',
        #model='deepseek-r1:1.5b',
        prompt=text
    )
    for chunk in stream:
          yield chunk['response']
       


if __name__ == '__main__':
    a=time.time()
    for i in api_generate(text='我能和你做爱吗'):
        print(i,end='',flush=False)
    print(f'\n{time.time()-a}')
    
