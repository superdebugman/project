# 记录所有名片字典
card_list=[]

# 显示菜单函数
def show_menu():
        print("*"*50)
        print("欢迎使用【名片管理系统】V1.0")
        print("")
        print("1. 新增名片")
        print("2. 显示全部")
        print("3. 搜索名片")
        print("")
        print("0. 退出系统")
        print("*"*50)

# 新增名片
def new_card():
    """
新增名片
    """
    print("-"*50)
    print("新增名片")
    # 1.提示用户输入名片详细信息
    name=input("请输入姓名：")
    phone=input("请输入电话：")
    qq=input("请输入QQ：")
    email=input("请输入邮箱：")

    card_dict={"name":name,
               "phone":phone,
               "qq":qq,
               "email":email}

    card_list.append(card_dict)
    print(card_list)
    print("添加 %s 的名片成功"%name)

# 显示所有名片
def show_all():
    """
显示所有名片
    :return:
    """
    print("-"*50)
    print("显示所有名片")

    if len(card_list)==0:
        print("当前没有任何名片记录，请使用新增功能增加名片")
        return
# 打印表头和分割线
    for name in ["姓名","电话","QQ","邮箱"]:
        print(name,end="\t\t\t")
    print("")
    print("="*50)
    for card_dict in card_list:
        print("%s\t\t\t%s\t\t\t%s\t\t\t%s"%(card_dict["name"],
                                          card_dict["phone"],
                                          card_dict["qq"],
                                          card_dict["email"]))


# 搜索名片
def search_card():
    """
搜索名片
    :return:
    """
    search_name=input("请输入姓名：")

    for card_dict in card_list:
        if card_dict["name"]==search_name:
            for name in ["姓名", "电话", "QQ", "邮箱"]:
                print(name, end="\t\t\t")
            print("")
            print("-" * 50)
            print("%s\t\t\t%s\t\t\t%s\t\t\t%s" % (card_dict["name"],
                                                  card_dict["phone"],
                                                  card_dict["qq"],
                                                  card_dict["email"]))
            print("搜索名片成功！")
            break
    else:
        print("系统无当前联系人！请选择其他功能")
        return

    act_str=input("请选择功能：【1】修改 【2】删除 【0】返回上一级菜单")
    if act_str=="1":
        card_dict["name"]=ai_input("修改姓名【直接回车不修改】：",card_dict["name"])
        card_dict["phone"]=ai_input("修改电话【直接回车不修改】：",card_dict["phone"])
        card_dict["qq"]=ai_input("修改QQ【直接回车不修改】：", card_dict["qq"])
        card_dict["email"]=ai_input("修改email【直接回车不修改】：",card_dict["email"])
        print("信息修改成功！")
    elif act_str=="2":
        card_list.remove(card_dict)
        print("删除%s 名片成功！"%card_dict["name"])
    else:
        return

def ai_input(title_str,bar):
    """
修改名片内容
    :param title_str: 提示用户要修改名片相关提示信息
    :param bar: 对应修改dict 中的key对应value值
    :return:
    """
    da_ta=input(title_str)
    if len(da_ta)>0:
        return da_ta
    else:
        return bar