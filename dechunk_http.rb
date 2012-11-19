#!/usr/bin/ruby

require 'optparse'

# Dechunks an HTTP chunked message body provided as input and writes the transformed message to output
#
def dechunk_http(input, output, verbose=false)
    @verbose = verbose
    chunk_header = ""
    num_chunks = 0

    def log(message)
        STDERR.puts message if @verbose
    end

    # Read the chunked input and write to the output
    begin
        while byte = input.readchar do
            if byte != 13 and byte != 10 
                chunk_header += byte.chr
                next
            elsif byte == 10
                chunk_bytes = chunk_header.hex
                log "parsed chunk header is #{chunk_header}"
                body = input.read(chunk_bytes)
                output.write body
                if input.readchar != 13 or input.readchar != 10
                    raise "expected chunk terminator, aborting"
                end
                log "wrote #{chunk_bytes} byte chunk body"
                num_chunks += 1
                if chunk_header == "0"
                    log "finished final chunk, consumed #{num_chunks} chunks"
                    return
                end
                chunk_header = ""
            end
        end
    rescue EOFError
        log "unexpected end of file, current chunk header #{chunk_header}"
        raise "unexpected end of file"
    end
end

if __FILE__ == $0
    SCRIPTNAME = File.basename(__FILE__)

    # Collect the command line options
    options = {}
    optparse = OptionParser.new do |opts|
        opts.banner = "Usage: #{SCRIPTNAME} [options] filename"
        options[:verbose] = false
        opts.on("-v", "--verbose", "Run verbosely") do |v|
            options[:verbose] = v
        end
        options[:outputfile] = "" 
        opts.on("-o", "--outfile FILE", "Write to an output file instead of stdout") do |o|
            options[:outputfile] = o
        end
    end
    begin
        optparse.parse!
    rescue Exception => e
        STDERR.puts "#{SCRIPTNAME}: error parsing options \'#{e}\'"
        puts optparse
        exit
    end

    # Get the input file
    if ARGV.empty?
        STDERR.puts "#{SCRIPTNAME}: input file required"
        puts optparse
        exit
    end

    unless File.exists?(ARGV[0])
        STDERR.puts "#{SCRIPTNAME}: input file '#{ARGV[0]}' not found"
        puts optparse
        exit
    end

    filename = ARGV[0]
    
    File.open(filename, 'r') do |input|
        if options[:outputfile] == ""
            dechunk_http(input, $stdout, options[:verbose])
        else
            File.open(options[:outputfile], 'w') do |output|
                dechunk_http(input, output, options[:verbose])
            end
        end
    end
end

