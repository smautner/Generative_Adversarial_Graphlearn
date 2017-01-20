import showroc as sr
import eden_tricks as et

def test_task_difficulty():
    X,y,graph=sr.get_data('1834')
    print et.task_difficulty(X,y)

test_task_difficulty()
