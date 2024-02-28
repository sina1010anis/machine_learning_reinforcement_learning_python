import pandas as pd
import numpy as np

address_file = 'Catchphish/CatchPhish_D3.csv'
address_new_file = 'Catchphish/CatchPhish_D3_My_Edit.csv'

def int_bool(string):
    return int(bool(string))

def dotInHost(string):
    string = string[8:]
    new_str = ''
    for i in string:
        new_str += i
        if i == '/':
            break

    return new_str

def catSubdomin(string):
    return string.split('.')[0].split('//')[1]

def catPath(string):
    # return len(string.split('/'))
    if len(string.split('/')) > 3:
        new_str = string.split('/')[3:]
        new_str = '/'.join(new_str).split('#')[0]
        return new_str
    else:
        return ''
    
    
def catParams(string):
    new_string = ''
    for i in range(len(string)):
        if string[i] == '#':
            new_string = string[i+1:]

    return new_string

data = pd.read_csv(address_file)

data_arr = np.array(data)

for i in range(len(data_arr)):

    url = data_arr[i][0]

    label = data_arr[i][1]
        
    #### At_url (2)
    data_arr[i][2] = int_bool(url.count('@'))


    ####  Amp_greater_equal (3)
    if url.count('&') > url.count('='):
        data_arr[i][3] = 1
    else:
        data_arr[i][3] = 0


    #### Delims_url (4)
    char = ['~', '`', '!', '^', '*', '(', ')', '[', ']', '{', '}', '"', "'", ';', ',', '>', '<', '|']
    for j in char:
        if int_bool(url.count(j)):
            data_arr[i][4] = 1
            break
        else:
            data_arr[i][4] = 0


    #### Other_delims_url (5)
    data_arr[i][5] = int(url.count('+')) + int(url.count('$')) +int(url.count('=')) +int(url.count('&')) +int(url.count(':')) +int(url.count('#')) +int(url.count('%')) 


    #### Len_url (6)
    data_arr[i][6] = len(url)


    #### Email_exist (7)
    data_arr[i][7] = 0


    #### Protocol_url (8)
    if int_bool(url.count('www.')) or int_bool(url.count('http:')):
        data_arr[i][8] = 1
    else:
        data_arr[i][8] = 0



    #### Suspwords_url (9)
    susp = ['server', 'client', 'confirm', 'account', 'banking', 'secure', 'ebayisapi', 'webscr', 'login', 'signin',' update', 'click', 'password', 'verify', 'lucky', 'bonus', 'suspend', 'paypal', 'wordpress', 'includes', 'admin', 'alibaba', 'myaccount', 'dropbox', 'themes', 'plugins', 'logout', 'signout', 'submit', ' limited', 'securewebsession', 'redirectme', 'recovery', 'secured', 'refund', 'webservis', 'giveaway', 'webspace', 'servico', 'webnode', 'dispute', 'review', 'browser', 'billing', 'temporary', 'restore', 'verification', 'required', 'resolution', '000webhostapp', 'webhostapp', 'wp', 'content', 'site', 'images', 'js', 'css', 'view']
    s = 0
    for ii in susp:
        if int_bool(url.count(ii)):
            s = s + int(url.count(ii))

    data_arr[i][9] = int(s)


    #### Digits_url (10)
    data_arr[i][10] = 0
    for iii in range(10):
        data_arr[i][10] = data_arr[i][10] + url.count(str(iii))


    #### Entropy_url (11)
    data_arr[i][11] = 0


    #### Tiny_url (12)
    data_arr[i][12] = 0



    #### Ratio_url_path (13)
    data_arr[i][13] = 0


    #### Dot_host (14)
    new_str = str(dotInHost(url))
    data_arr[i][14] = new_str.count('.')



    #### Len_subdomain (15)
    data_arr[i][15] = len(catSubdomin(url))


    #### Having_https (16)
    if int_bool(url.count('https:')):
        data_arr[i][16] = 1
    else:
        data_arr[i][16] = 0


    #### Brand_host (17)
    data_arr[i][17] = 0


    #### Host_large_tok (18)
    data_arr[i][18] = 0


    #### Path_large_tok (19)
    data_arr[i][19] = 0


    #### TLD_path (20)
    data_arr[i][20] = 0


    #### Len_fle (21)
    data_arr[i][21] = 0  


    #### Extension (22)
    data_arr[i][22] = 0  


    #### Delims_params (23)
    char = ['~', '`', '!', '^', '*', '(', ')', '[', ']', '{', '}', '"', "'", ';', ',', '>', '<', '|']
    for j in char:
        if int_bool(catPath(url).count(j)):
            data_arr[i][23] = 1
            break
        else:
            data_arr[i][23] = 0 


    #### Len_params (24)
    data_arr[i][24] = len(catParams(url)) 



    #### Dot_path (25)
    data_arr[i][25] = catPath(url).count('.')


    #### Hyphen_path (26)
    data_arr[i][26] = catPath(url).count('-')



    #### Slash_path (27)
    data_arr[i][27] = catPath(url).count('/')


    #### Len_path (28)
    data_arr[i][28] = len(catPath(url))


    #### Brand_path (29)
    data_arr[i][29] = 0


    #### Digits_path (30)
    data_arr[i][30] = catPath(url).count('0')+catPath(url).count('1')+catPath(url).count('2')+catPath(url).count('3')+catPath(url).count('4')+catPath(url).count('5')+catPath(url).count('6')+catPath(url).count('7')+catPath(url).count('8')+catPath(url).count('9')



    #### Entropy_path (31)
    data_arr[i][31] = 0

    
data = pd.DataFrame(data_arr, columns=data.columns, index=data.index)

data.to_csv(address_new_file)

print('\n', data)








# for i in np.arange(1, len(data)):
#     print(i,'\n')