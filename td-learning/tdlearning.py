# Temporal differencel learning algorithm with reward function customization
# Notes: 1- Link mats: possible actions, main diagonal elements should be 0 
#           this will eliminate to visit itself.
#        2- np.random.randint generates rand nums from uniform distribution
#           (np.random.rand(nactions, nstates)*0.1-0.5)
#        3- In each iteration if selected s,a pair is 0 in links, re-select the a.

import numpy as np
import termcolor as tcolor
import seaborn as sns

def main():

    print tcolor.colored("===========TOP-DOWN MAIN=========", 'red')
    print tcolor.colored("0) TD Learning starts", 'green')
    print tcolor.colored("1) TD Learning parameters", 'green') 
    print tcolor.colored("   1.1) gamma, mu, epsilon, iterations", 'cyan')
    mu, gamma, epsilon, nits, iters = 0.7, 0.4, 0.1, 0, 10
    nactions, nstates = 20, 20

    print tcolor.colored("   1.2) matrix creation: Reward, Links, Q", 'cyan')
    links = np.ones((nstates, nstates),dtype=int)
    np.fill_diagonal(links, 0) # set main diagonal elements as zero
    # to put negative values on the qmat use -0.5
    q_mat = np.random.rand(nactions, nstates)*0.1-0.5

    print tcolor.colored("2) TD Starts", 'green') 
    print tcolor.colored("   2.1) select a random state", 'cyan') 
    print tcolor.colored("   2.2) generate random epsilon threshold", 'cyan') 
    print tcolor.colored("   2.3) select an action based on policy", 'cyan') 
    print tcolor.colored("        2.3.1) either follow policy of expoit environment", 'cyan')
    print tcolor.colored("   2.4) there is no end state, iterate nstates times over Qmat", 'cyan') 
    print tcolor.colored("   2.5) extract reward value for s,a pair", 'cyan')
    print tcolor.colored("   2.6) set next state as current action", 'cyan')
    print tcolor.colored("   2.7) update Q(s,a) value", 'cyan')
    print tcolor.colored("   2.8) display Q(s,a) heatmaps", 'cyan')
    print tcolor.colored("3) Display final Q value heatmap", 'green') 

    sns.plt.figure()    

    # temporal reward values for quick and dirty demo
    tmp_reward = [-3, 10]
    visits = np.zeros((nstates, nactions), dtype=int)

    while nits < iters:
        s = np.random.randint(nstates) # state
        eps = np.random.rand() # epsilon threshold

        if eps < epsilon:
            # extract non-zero elements of links
            nonz =  np.where(links[s,:]!=0)
            rindx =np.random.randint(np.shape(nonz)[1])
            a = nonz[0][rindx]
        else:
            a = np.argmax(q_mat[s,:])

        nits2 = 0
        while nits2 < nstates:
            rew = np.random.randint(2)
            sdot = a
            
            eps2 = np.random.rand()
            if eps2 < epsilon:
                nonz2 = np.where(links[sdot,:]!=0)
                rindx2 =np.random.randint(np.shape(nonz2)[1])
                adot = nonz2[0][rindx2]
            else:
                adot = np.argmax(q_mat[sdot,:])
                # print "next state: ", sdot, "next action: ", adot

            q_mat[s,a] += mu * (rew + gamma*q_mat[sdot,adot]-q_mat[s,a])

            s = sdot
            a = adot
            # log number of visits to extract valuable s,a pairs
            visits[s,a] += 1
            
            sns.heatmap(q_mat)
            sns.plt.show(block=False)
            sns.plt.clf()

            nits2 += 1
        nits += 1

    sns.plt.close()

    sns.plt.figure(2)
    sns.heatmap(q_mat)
    sns.plt.title("Q Matrix Heatmap")
    sns.plt.ylabel("States")
    sns.plt.xlabel("Next states")

    sns.plt.figure(3)

    # instead of color put number of visits
    sns.heatmap(visits, annot=True, fmt="d", cmap="YlGnBu")
    sns.plt.title("s,a pairs visit Heatmap")
    sns.plt.ylabel("States")
    sns.plt.xlabel("Next states")
    sns.plt.show(block=True)

if __name__ == '__main__':
    main()
