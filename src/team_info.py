# 定義 MLB 架構 (使用資料集中的 3 碼隊名)
MLB_STRUCTURE = {
    'AL': {
        'East': ['BAL', 'BOS', 'NYA', 'TBA', 'TOR'],
        'Central': ['CHA', 'CLE', 'DET', 'MIN', 'KCA'],
        'West': ['HOU', 'ANA', 'OAK', 'SEA', 'TEX']
    },
    'NL': {
        'East': ['ATL', 'MIA', 'NYN', 'PHI', 'WAS'],
        'Central': ['CHN', 'CIN', 'MIL', 'PIT', 'SLN'],
        'West': ['ARI', 'SFN', 'LAN', 'COL', 'SDN']
    }
}

# 建立反查字典: Team -> "League Division" (例如 NYA -> "AL East")
TEAM_DISPLAY_INFO = {}
for league, divisions in MLB_STRUCTURE.items():
    for division, teams in divisions.items():
        for team in teams:
            TEAM_DISPLAY_INFO[team] = f"{league} {division}"

def get_team_league(team_code):
    """回傳球隊所屬聯盟 (AL 或 NL)"""
    info = TEAM_DISPLAY_INFO.get(team_code, "")
    return info.split(" ")[0] if info else "Unknown"

def get_team_division(team_code):
    """回傳球隊所屬分區 (例如 AL East)"""
    return TEAM_DISPLAY_INFO.get(team_code, "Unknown")

def get_relation(team1, team2):
    """
    判斷兩隊的關係，用於決定賽程權重
    回傳:
    - 'DIVISION': 同聯盟且同分區 (打最多場)
    - 'LEAGUE': 同聯盟但不同分區 (打次多場)
    - 'INTER': 不同聯盟 (跨聯盟，打最少場)
    """
    info1 = TEAM_DISPLAY_INFO.get(team1, "")
    info2 = TEAM_DISPLAY_INFO.get(team2, "")
    
    # 如果找不到隊伍資訊，預設為跨聯盟
    if not info1 or not info2:
        return 'INTER'
    
    parts1 = info1.split() # ['AL', 'East']
    parts2 = info2.split() # ['AL', 'Central']
    
    league1, div1 = parts1[0], parts1[1]
    league2, div2 = parts2[0], parts2[1]
    
    if league1 != league2:
        return 'INTER'     # 不同聯盟
    elif div1 == div2:
        return 'DIVISION'  # 同聯盟且同分區
    else:
        return 'LEAGUE'    # 同聯盟不同分區