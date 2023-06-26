class Session:
    """

    """

    def __init__(self, timestamp, clicked_news, ignored_news, shown_news):
        """

        :param timestamp:
        :param shown_news:
        :param clicked_news:
        :param ignored_news:
        """
        self.timestamp = timestamp
        self.clicked_news = clicked_news
        self.ignored_news = ignored_news
        self.shown_news = shown_news

    def __str__(self):
        info = f"{self.timestamp}\n"
        info += "\tClicked News\n"
        info += f"\t\t{self.clicked_news}\n"
        info += "\tIgnored News\n"
        info += f"\t\t{self.ignored_news}\n"
        info += "\tShown News\n"
        info += f"\t\t{self.shown_news}\n"
        info += "-----------\n"
        return info

class UserSessions:
    """

    """

    def __init__(self, user_id, sessions=None):
        self.user_id = user_id
        self.sessions = [] if sessions is None else sessions

    def add_session(self, session):
        self.sessions.append(session)

    def __str__(self):
        info = ""
        info += f"User: {self.user_id}\n-----------\n"
        for session in self.sessions:
            info += str(session)
        return info


def get_user_sessions(user_id, data_behaviors):
    user_sessions = UserSessions(user_id)
    session_cols = ["timestamp", "clicked_news", "ignored_news", "shown_news"]
    for impression in data_behaviors[data_behaviors["user_id"] == user_id][session_cols].values:
        session = Session(*impression)
        user_sessions.add_session(session)
    return user_sessions
