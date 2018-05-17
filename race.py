import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from maps import *

#行動
def f(x,y,p_f):
    action = np.random.choice(8,p=p_f[y,x])
    rand_x = np.random.choice([-1,0,1],p=[0,1,0])
    rand_y = np.random.choice([-1,0,1],p=[0,1,0])
    return action, rand_x+(action+1)%3, rand_y+(action+1)//3

def f_first(p_f,s_num):
    action = np.random.choice(s_num,p=p_f)
    return action

def mc():
    #初期化
    s = s0_0[::-1,:]
    r,c = s.shape
    s_pad = np.pad(s,(3,3),'constant',constant_values=1)
    w_r,w_c = np.where(s==1)
    eps = 0.1
    search_break_flag = 0
    search_break_cnt = 0

    #描写
    fig = plt.figure()
    ims = []
    im_global = []

    #方策pの初期化
    p = np.ones((r,c,8))*(1/8)
    p_first = np.ones((np.sum(s==2)))*(1/np.sum(s==2))
    r_s,c_s = np.where(s==2)

    #行動価値、選択回数の初期化
    q = np.zeros((r,c,8))
    n = np.zeros((r,c,8))
    q_first = np.zeros((np.sum(s==2)))
    n_first = np.zeros((np.sum(s==2)))
    Rs=np.array((0))
    Rss = np.array(())
    Rss_all = np.array(())

    for j in range(1000):
        #エピソード生成 
        x0,y0 = 0,0
        x1,y1 = 0,0
        ars = np.array([0,0,0,0],ndmin=2)
        #探索中
        search = 1
        search_break_flag = 0

        #移動処理
        while(search):
            if ars.shape[0] == 1:
                act_first = f_first(p_first,c_s.shape[0])
                x0 = c_s[act_first]
                y0 = r_s[act_first]
                ars[0,0] = act_first
                ars[0,2] = x0
                ars[0,3] = y0

            act,a_x,a_y= f(x0,y0,p)
            reward = -1
            a_yy = np.arange(abs(a_y)+1)
            a_yy = np.where(a_y>=0,a_yy,a_yy*-1)
            a_xx = np.arange(abs(a_x)+1)
            a_xx = np.where(a_x>=0,a_xx,a_xx*-1)
            for i in a_yy:
                if s_pad[y0+i+3,x0+3]==1: #1:wall
                    y1 = y0+a_yy[i-1] #a_yy[0]で元の場所なのでi-1>=0
                    reward = -5
                    break
                elif s_pad[y0+i+3,x0+3]==3: #3:goal
                    y1 = y0+a_yy[i]
                    search = 0
                    break
            else:
                y1 = y0 + a_y

            for i in a_xx:
                if s_pad[y0+3,x0+i+3]==1:
                    x1 = x0+a_xx[i-1]
                    reward = -5
                    break
                elif s_pad[y0+3,x0+i+3]==3:
                    x1 = x0+a_xx[i]
                    search = 0
                    break
            else:
                x1 = x0 + a_x

            if ((x0==x1)&(y0==y1)):
                x_next = s_pad[y0+3,x0+4]
                y_next = s_pad[y0+4,x0+3]
                if x_next != 1:
                    x1 += 1
                elif y_next != 1:
                    y1 += 1

            x0,y0 = x1,y1
            ars = np.append(ars,[[act,reward,x0,y0]],axis=0)

        #行動価値更新
        if search_break_flag == 0:
            #print(ars)
            n[ars[:,3],ars[:,2],ars[:,0]] += 1
            Rs = (ars[:,1])[::-1].cumsum()[::-1]
            q[ars[:,3],ars[:,2],ars[:,0]] = (Rs + (n[ars[:,3],ars[:,2],ars[:,0]]-1)*q[ars[:,3],ars[:,2],ars[:,0]]) / (n[ars[:,3],ars[:,2],ars[:,0]])
            n_first[ars[0,0]] += 1
            q_first[ars[0,0]] = (Rs[0] + (n_first[ars[0,0]]-1)*q_first[ars[0,0]]) / (n_first[ars[0,0]])

            #update policy
            p = np.ones((r,c,8))*(eps/8)
            idx_qmax = q.argmax(axis=2)
            idx_0,idx_1 = np.ogrid[:r,:c]
            p[idx_0,idx_1,idx_qmax] = 1 -((7/8)*eps)
            Rss = np.append(Rss, Rs.min())
            Rss_all = np.append(Rss_all,Rs.min())

            p_first = np.ones(p_first.shape)*(eps/p_first.shape[0])
            p_first[q_first.argmax()] = 1 - (eps*(p_first.shape[0]-1)/(p_first.shape[0]))

        #描写
        #depend on map
        im = plt.plot(ars[:,2],ars[:,3],color='black')
        bar_h = (Rs.min()+90)*(1/6)
        m1 = 100
        if Rss_all.shape[0] < 100: m1 =Rss_all.shape[0]
        bar_avr = (Rss_all[-m1:].mean()+90)*(1/6)
        r_bar = plt.fill([13,13.5,13.5,13],[1,1,bar_h,bar_h],color='r')
        avr_bar = plt.fill([15,15.5,15.5,15],[1,1,bar_avr,bar_avr],color='b')
        text0 = plt.text(13,0,'R')
        text1 = plt.text(14,0,'average')
        text2 = plt.text(7.5,0,'episode%s'% (j+1))
        text3 = plt.text(7.5,1,' %s'%Rs.min(),fontsize=15)
        text4 = plt.text(7.5,1.8,'R:')
        wall1 = plt.plot([0,0,2,2,5,5,17],[0,8,8,13,13,18,18],'g-',linewidth=5)
        wall2 = plt.plot([5,5,10,10,17],[0,7,7,14,14],'g-',linewidth=5)
        text5 = plt.text(2,0,'start')
        text6 = plt.text(16,13,'goal')
        ims.append(im+[text0]+[text1]+[text2]+[text3]+[text4]+r_bar+avr_bar+wall1+wall2+[text5]+[text6])

    plt.axis('scaled')
    plt.xlim(-0.5,c-0.5)
    plt.ylim(-0.5,r+1)
    plt.title('Monte Carlo method')
    ani = animation.ArtistAnimation(fig,ims)
    ani.save('anime_all.mp4',writer='ffmpeg',fps=10)
    plt.show()

if __name__ == '__main__':
    mc()
