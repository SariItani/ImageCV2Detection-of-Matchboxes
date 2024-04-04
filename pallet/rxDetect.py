import threading
import time
import socketserver as ss
import wave
import sys
import getopt

cli_addr = '0.0.0.0'
port = 19810
chunk_size = 1024
sleep_time = 100
ACK_MSG = "DONE"       # confirmation message for robot action complete
EMPTY_MSG = "EMPTY"    # message to notify empty pallet to be replaced
robot_action_time = 5  # simulates the time needed to robot for bag removal

class ThreadedTCPRequestHandler(ss.BaseRequestHandler):

    def handle(self):
        cur_thread = threading.current_thread()
        if verbose: print('Connecting {}'.format(self.client_address[0]))

        timestr = time.strftime("%Y%m%d-%H%M%S")
        data = self.request.recv(chunk_size)
        # create a list of bag centers ordered according to
        # the box area; box list is ordered on the client device
        msg = data.decode('utf-8')
        if msg == EMPTY_MSG:
            print("EMPTY PALLET: please replace it.")
            input("Press ENTER after replacement.")
        else:
            centers = list(tuple(msg.split('-')))
            print(centers)
            # the next line mimics time spent by robot to select a bag
            time.sleep(robot_action_time)

        self.request.sendall(ACK_MSG.encode('utf-8'))
        if verbose: print('Disconnecting {}'.format(self.client_address[0]))

class ThreadedTCPServer(ss.ThreadingMixIn, ss.TCPServer):
    pass

if __name__ == "__main__":

    verbose = False

    def usage():
        print("usage: {} -h -v -p server_port".format(sys.argv[0]))
        sys.exit(2)

    try:
        options, arguments = getopt.getopt(sys.argv[1:],'hvp:',['help','verbose','port='])
        #assert len(sys.argv) > 1
    except:
        usage()

    for opt, arg in options:
        if opt == '-h':
            usage()
        elif opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-p", "--port"):
            port = int(arg)
        else:
            usage()

    server = ThreadedTCPServer((cli_addr, port), ThreadedTCPRequestHandler)
    server.allow_reuse_address = True
    
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    try:
        server_thread.start()
        if verbose: print("Server started at {} on port {}".format(cli_addr, port))
        server_thread.join()
        while True: time.sleep(sleep_time)
    except (KeyboardInterrupt, SystemExit):
        server.shutdown()
        server.server_close()
        sys.exit(0)
