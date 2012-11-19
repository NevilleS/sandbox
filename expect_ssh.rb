#!/usr/bin/ruby

require 'optparse'
require 'rubygems'
require 'net/ssh'

# Executes a sequence of commands in a shell on a remote host. Yields control of the channel an optionally provided
# block after executing all the given commands, if you want to send some more commands. Returns a single string of all
# output generated by the remote host
#
def expect_ssh(host, user, password, commands, verbose=false, opts={})
    @verbose = verbose

    def log(message)
        STDERR.puts message if @verbose
    end

    output = ""
    opts[:password] = password if opts[:password].nil?
    Net::SSH.start(host, user, opts) do |ssh|
        log "started session"
        channel = ssh.open_channel do |channel, success|
            log "opened channel"
            channel.send_channel_request "shell" do |ch, success|
                raise "couldn't start shell" unless success
                ch.on_data do |ch, data|
                    log "received data: #{data}"
                    output << data
                end
                commands.each do |command|
                    log "sending command: #{command}"
                    ch.send_data "#{command}\n"
                end
                yield channel if block_given?
            end
        end
        ssh.loop
    end
    return output
end

if __FILE__ == $0
    SCRIPTNAME = File.basename(__FILE__)

    # Collect the command line options
    options = {}
    optparse = OptionParser.new do |opts|
        opts.banner = "Usage: #{SCRIPTNAME} [options] commands"
        options[:verbose] = false
        opts.on("-v", "--verbose", "Run verbosely") do |v|
            options[:verbose] = v
        end
        options[:dontexit] = false
        opts.on("-x", "--dontexit", "Don't automatically append an 'exit' command to the provided commands") do |x|
            options[:dontexit] = x
        end
        opts.on("-o", "--outfile FILE", "Write to an output file instead of stdout") do |o|
            options[:outfile] = o
        end
        opts.on("-h", "--host HOST", "remote host (e.g. myserver.com)") do |h|
            options[:host] = h
        end
        opts.on("-u", "--user USER", "username (e.g. admin)") do |u|
            options[:user] = u
        end
        opts.on("-p", "--password PASSWORD", "password for USER@HOST") do |p|
            options[:password] = p
        end
    end
    begin
        optparse.parse!
        missing = [:host, :user, :password].select { |opt| options[opt].nil? }
        raise "missing required options: #{missing.join(', ')}" unless missing.empty?
    rescue Exception => e
        STDERR.puts "#{SCRIPTNAME}: error parsing options \'#{e}\'"
        puts optparse
        exit
    end

    if ARGV.empty?
        STDERR.puts "#{SCRIPTNAME}: specify one or more commands"
        puts optparse
        exit
    end

    commands = ARGV
    commands << "exit" unless options[:dontexit]

    if options[:verbose]
        STDERR.puts "attempting to connect #{options[:user]}@#{options[:host]} and execute the following commands:"
        ARGV.each_with_index do |command, i|
            STDERR.puts "#{i+1}:    #{command}"
        end
    end

    begin
        output = expect_ssh(options[:host], options[:user], options[:password], commands, options[:verbose])
        STDERR.puts "retrieved output, writing to stdout or outfile" if options[:verbose]
        if options[:outfile].nil?
            puts output
        else
            File.open(options[:outfile], 'w') do |outfile|
                outfile.write output
            end
        end
    rescue Exception => e
        STDERR.puts "#{SCRIPTNAME}: error during execution \'#{e}\'"
        puts optparse
        exit
    end
end
