import pymysql

class cMember :
    def __init__(self, member_id = None, member_pw = None, name = None, mobile = None, email = None) :
        self.member_id = member_id
        self.member_pw = member_pw
        self.name = name
        self.mobile = mobile
        self.email = email
        

class cMember_dao :
    def __init__(self) :
        self.conn = None
        self.cur = None

    def Connect(self) :
        self.conn = pymysql.connect(host = "3.37.132.145", user = "team4", password = "team4",
                                    db = "project", charset = "utf8")
        self.cur = self.conn.cursor()

    def Disconnect(self) :
        self.cur = None
        self.conn.close()

    def Insert(self, vo) :
        self.Connect()
        sql = "insert into member(member_id, member_pw, name, mobile, email) values(%s, %s, %s, %s, %s)"
        vals = (vo.member_id, vo.member_pw, vo.name, vo.mobile, vo.email)
        self.cur.execute(sql, vals)
        self.conn.commit()
        self.Disconnect()

    def SelectAll(self) :
        self.Connect()
        sql = "select * from member"
        self.cur.execute(sql)
        mem = []
        for row in self.cur :
            mem.append(cMember(row[0], row[1], row[2], row[3], row[4]))
        self.Disconnect()
        return mem

    def Select(self, email) :
        self.Connect()
        sql = "select * from member where email = %s"
        vals =(email,)
        self.cur.execute(sql, vals)
        row = self.cur.fetchone()
        self.Disconnect()
        if row :
            return cMember(row[0], row[1], row[2], row[3], row[4])
   
    def Select(self, member_id, member_pw) :
        self.Connect()
        sql = "select * from member where member_id = %s"
        vals =(member_id,)
        self.cur.execute(sql, vals)
        row = self.cur.fetchone()
        self.Disconnect()
        if row and row[1] == member_pw :
            return True
        return False
            
    def SelectOne(self, member_id) :
        self.Connect()
        sql = "select * from member where member_id = %s"
        vals =(member_id,)
        self.cur.execute(sql, vals)
        row = self.cur.fetchone()
        self.Disconnect()
        if row :
            return cMember(row[0], row[1], row[2], row[3], row[4])
       

    def Update(self, new_pw, member_id) :
        self.Connect()
        sql = "update member set member_pw = %s where member_id = %s"
        vals = (new_pw, member_id)
        self.cur.execute(sql, vals)
        self.conn.commit()
        self.Disconnect()
        
    def Update_info(self, new_name, new_mobile, member_id) :
        self.Connect()
        sql = "update member set name = %s, mobile = %s where member_id = %s"
        vals = (new_name, new_mobile, member_id)
        self.cur.execute(sql, vals)
        self.conn.commit()
        self.Disconnect()

        
        
        
class cMember_service :
    def __init__(self) :
        self.dao = cMember_dao()

    def Add_member(self, vo) :
        return self.dao.Insert(vo)

    def Get_all(self) :
        return self.dao.SelectAll()

    def Get_member(self, email) :
        return self.dao.Select(email)
    
    def Get_member_by_id(self, member_id) :
        return self.dao.SelectOne(member_id)
    
    def Check_id(self, member_id, member_pw) :
        return self.dao.Select(member_id, member_pw)

    def Edit_member(self, new_pw, member_id) :
        return self.dao.Update(new_pw, member_id)
    
    def Edit_member_info(self, new_name, new_mobile, member_id) :
        return self.dao.Update_info(new_name, new_mobile, member_id)

    
        