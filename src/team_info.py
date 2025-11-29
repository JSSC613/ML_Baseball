# src/team_info.py
TEAM_NAME_MAP_CN = {'BAL': '巴爾的摩金鶯', 'BOS': '波士頓紅襪',
                    'NYA': '紐約洋基隊', 'TBA': '坦塔灣光芒',
                    'TOR': '多倫多藍鳥', 'CHA': '芝加哥白襪',
                    'CLE': '克里夫守護者', 'DET': '底特律老虎',
                    'MIN': '明尼蘇達雙城', 'KCA': '堪薩斯皇家', 
                    'HOU': '休士頓太空人隊', 'ANA': '洛杉磯天使', 
                    'OAK': '奧克蘭運動家', 'SEA': '西雅圖水手', 
                    'TEX': '德州遊騎兵', 'ATL': '亞特蘭大勇士', 
                    'MIA': '邁阿密馬林魚', 'NYN': '紐約大都會', 
                    'PHI': '費城人', 'WAS': '華盛頓國民', 
                    'CHN': '芝加哥小熊', 'CIN': '辛辛那提', 
                    'MIL': '密爾瓦基釀酒人', 'PIT': '匹茲堡海盜', 
                    'SLN': '聖路易紅雀', 'ARI': '亞利桑那響尾蛇', 
                    'SFN': '舊金山巨人', 'LAN': '洛杉磯道奇', 
                    'COL': '科羅拉多落機', 'SDN': '聖地牙哥教士' }
# 定義 MLB 架構
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

# 建立反查字典
TEAM_DISPLAY_INFO = {}
for league, divisions in MLB_STRUCTURE.items():
    for division, teams in divisions.items():
        for team in teams:
            TEAM_DISPLAY_INFO[team] = f"{league} {division}"
TEAM_TO_DIV = {}
for league, divisions in MLB_STRUCTURE.items():
    for division, team_codes in divisions.items():
        for code in team_codes:
            TEAM_TO_DIV[code] = f"{league} {division}"
def get_relation(team1_code, team2_code):
    info1 = TEAM_TO_DIV.get(team1_code, "")
    info2 = TEAM_TO_DIV.get(team2_code, "")

    if not info1 or not info2:
        return "INTER"

    L1, D1 = info1.split()
    L2, D2 = info2.split()

    if L1 != L2:
        return "INTER"
    if D1 == D2:
        return "DIVISION"
    return "LEAGUE"
def get_team_list():
    return [
        {"code": code, "display": cname}
        for code, cname in TEAM_NAME_MAP_CN.items()
    ]