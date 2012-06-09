#! /usr/local/bin/ruby
# Ruby module that implements algorithms for MTE204: Numerical Methods
# Author: Neville Samuell

module NumMethods

    # Encapsulates a matrix, which is just a 2D array organized by row then column
    class Matrix
        attr_accessor :data
        attr_reader :height, :width
        def initialize(data=0.0, size=1,title=nil)
            if data.is_a?(Array)
                @data = data
            else
                if size.is_a?(Array)
                    @data = Array.new(size[0]) do 
                        Array.new(size[1], data)
                    end
                else
                    @data = Array.new(size) do
                        Array.new(size, data)
                    end
                end
            end
            @height = @data.size
            @width = @data[0].size
            self.each { |i,j| self[i,j] *= 1.0 }
            @data
        end

        # Alternate constructor
        def Matrix::diagonal(value,size=1)
            raise "Diagonal matrices must be square" if size.is_a?(Array) and size[0]!=size[1]
            diag = Matrix.new(0,size)
            diag.each_row do |i,row|
                row[i] = value
            end
            diag
        end

        # Column vector constructor, from an array
        def Matrix::column_vector(data)
                row_data = []
                data.size.times { |i| row_data[i] = [data[i]] }
                Matrix::new(row_data)
        end

        # Construct a Matrix from a file
        def Matrix::from_file(path,to_f=true,seperator=/,/,remove=/\s/)
            File.open(path) do |file|
                data = []
                file.each_line do |line|
                    terms = line.chomp.gsub(remove,"").split(seperator)
                    terms.collect! { |term| term.to_f } if to_f
                    data << terms unless terms.nil?
                end
                Matrix.new(data)
            end
        end

        # Write a matrix to a file
        def to_file(path,title=nil,seperator=", ")
            File.open(path,"w+") do  |file|
                file.write("#{title}\n") unless title.nil?
                self.height.times do |i|
                    (self.width-1).times do |j|
                        file.write("#{self[i,j]}#{seperator}")
                    end
                    file.write("#{self[i,self.width-1]}\n")
                end
            end 
        end

        # Print the matrix
        def print(title=nil)
            strings = [] 
            max_lengths = Array.new(self.width,0)
            self.height.times do |i|
                strings[i] = []
                self.width.times do |j|
                    string = self[i,j].to_s
                    max_lengths[j] = if string.length > max_lengths[j] then string.length else max_lengths[j] end
                    strings[i].push(string)
                end
            end
            puts "#{title}" unless title.nil?
            self.height.times do |i|
                self.width.times do |j|
                    extra = 1 + max_lengths[j] - strings[i][j].length
                    Kernel.print(strings[i][j] + " "*extra)
                end
                Kernel.print "\n"
            end
            self
        end

        # Use the Gruff gem to output the Matrix as an image
        # This method assumes the Matrix is a collection of points where col(0) is the x values and each
        # column contains y values for a different set of data
        def to_graph(title,names,filename,xstep=1,axes=["x","y"],graphsize=800)
            require "rubygems"
            require "gruff"

            graph = Gruff::Line.new(graphsize)
            1.upto(width-1) { |j| graph.data(names[j-1], self.col(j)) }
            # Generate x-axis labels
            if xstep != 0
                xlabels = {}
                0.step(self.height-1,xstep) do |i|
                    xlabels[i] = self[i,0].to_s
                end
            end
            graph.labels= xlabels
            graph.title = title
            graph.x_axis_label = axes[0]
            graph.y_axis_label = axes[1]
            graph.write(filename)
        end

        # Returns a smaller version of self
        def truncate(new_size)
            a = Matrix.new(0,new_size)
            a.each { |i,j,val| a[i,j] = self[i,j] }
            a
        end
       
        # Returns an identical copy of this matrices values
        def copy
            cpy = Matrix.new(0,self.size)
            cpy.each { |i,j,val| cpy[i,j] = self[i,j] }
            cpy
        end

        def square?
            size[0] == size[1]
        end

        def size
            [@height, @width]
        end

        def row(i)
            @data[i]
        end

        def col(j)
            col = Array.new(0,@height)
            @height.times { |i| col[i] = self[i,j] }
            col
        end

        def []=(i,j,value)
            raise "Index out of bounds" unless i < @height and j < @width
            data[i][j]=value*1.0
        end

        def [](i,j=0)
            data[i][j]
        end

        def ==(another)
            return false unless another.is_a?(Matrix)
            return false unless another.size == self.size
            self.each { |i,j| return false unless self[i,j] == another[i,j] }
            true
        end

        def +(another)
            raise "Can only add a matrix to another matrix" unless another.is_a?(Matrix)
            raise "Cannot add different sized matrices together" if self.size != another.size
            sum = Matrix.new(0, self.size)
            (@height).times { |i| (@width).times { |j| sum[i,j] = self[i,j] + another[i,j] } }
            sum
        end

        def -(another)
            raise "Can only subtract a matrix from another matrix" unless another.is_a?(Matrix)
            raise "Cannot subtract different sized matrices" if self.size != another.size
            diff = Matrix.new(0, self.size)
            (@height).times { |i| (@width).times { |j| diff[i,j] = self[i,j] - another[i,j] } }
            diff
        end

        def *(another)
            if another.is_a?(Numeric)
                product = Matrix.new(0, self.size)
                (self.height).times { |i| (self.width).times { |j| product[i,j] = self[i,j]*another } }
                product
            else
                if self.width != another.height
                    raise "Cannot multiply a #{self.height}x#{self.width} matrix by a #{another.height}x#{another.width} matrix"
                end
                product = Matrix.new(0, [self.height, another.width])
                (self.height).times do |i|
                    (another.width).times do |j|
                        (self.width).times do |k|
                            product[i,j] += self[i,k]*another[k,j]
                        end
                    end
                end
                product
            end
        end

        def /(another)
            if another.is_a?(Numeric)
                self*(1.0/another)
            else
                raise "Do not know how to divide matrices"
            end
        end

        # Traverse the matrix in row order, ascending through columns
        def each
            (@height).times do |i|
                (@width).times do |j|
                    yield(i,j,@data[i][j])
                end
            end
            self
        end

        def each_row
            (@height).times do |i|
                yield(i,@data[i])
            end
            self
        end

    end

    # Generates a proc object to evaluate a polynomial using a column vector of coefficients in increasing order of x
    def NumMethods::polynomial_from_matrix(coeffs)
        raise "coeffs must be a column vector" unless coeffs.width == 1
        Proc.new do |x|
            sum = 0
            coeffs.height.times do |i|
                sum += coeffs[i]*(x**i)
            end
            sum
        end
    end

    # Helper function to validate that the matrices in a matrix equation Ax = b are valid sizes
    def NumMethods::valid_equation?(a,x,b)
        unless a.square? and a.height == x.height and x.size == b.size and b.width == 1
           raise "Invalid matrix equation"
        end
        true
    end

    # Integer base converter
    # Converts the input string from the input base to the output base and returns it as a string
    def NumMethods::convert_base(input_str, input_base, output_base)
        # Convert the string to an array of decimal values in the input base
        input_str.upcase!
        input = []
        input_str.each_byte do |byte|
            input.push(if byte >= 64 then byte-55 else byte-48 end)
        end
        # Sum the array to create a decimal integer
        sum = 0
        input.each_index do |i| 
            sum += input[i]*(input_base**(input.size-1-i));
            raise "Input string contains characters beyond the input base" if input[i] > (input_base - 1)
        end
        # Convert the decimal integer into an array of decimal values in the output base
        output = []
        i = 0
        until sum <= 0
            i += 1
            num = sum%(output_base**i)
            div = output_base**(i-1)
            output.insert(0,num/div)
            sum -= num
        end
        # Convert the decimal values into strings (a,b,etc)
        output_str = ""
        output.each { |val| output_str.concat(if val < 10 then val+48 else val+55 end) }
        output_str
    end

    # Integer base converter
    # Converts a matrix where each row is [input_str, input_base, output_base] into a new column vector
    # of strings in the output base
    def NumMethods::convert_base_matrix(a)
        output = Matrix.new("",[a.height,1])
        a.each_row do |i,row|
            output[i,0] = convert_base(row[0],row[1].to_i,row[2].to_i)
        end
        output
    end

    # Gauss-Siedel solver
    # Solves a matrix equation of the form ax = b
    # Requires an initial guess and an acceptable relative error(0<rel_error<1) and (optionally) absolute error
    def NumMethods::gauss_siedel_solve(a,guess,b,rel_error,abs_error=Float::MAX)
        x = guess.copy
        NumMethods::valid_equation?(a,x,b)
        current_rel_error = 1.0
        current_abs_error = Float::MAX-1.0e-100
        until current_rel_error < rel_error and current_abs_error < abs_error
            rel_errors,abs_errors = gauss_siedel_iterate!(a,x,b)
            current_rel_error = rel_errors.max
            current_abs_error = abs_errors.max
        end
        x
    end

    # Gauss-Siedel iteration
    # Performs a single update of the current solution and returns the relative and absolute errors
    def NumMethods::gauss_siedel_iterate!(a,current_soln,b)
        rel_errors = []
        abs_errors = []
        # Update each value
        a.each_row do |i,row|
            next_soln = b[i]
            row.size.times do |j|
                next_soln -= a[i,j]*current_soln[j] unless i==j
            end
            next_soln /= a[i,i]
            rel_errors[i] = ((next_soln - current_soln[i])/next_soln).abs
            abs_errors[i] = (next_soln - current_soln[i]).abs
            current_soln[i,0] = next_soln
        end
        return rel_errors,abs_errors
    end

    # Gauss Elimination with Partial Pivoting
    # Solves a matrix equation of the form ax = b
    # Rearranges the order of equations to avoid error
    def NumMethods::gauss_partial_pivot_solve(a,b)
        x = NumMethods::Matrix.new(0, b.size)
        NumMethods::valid_equation?(a,x,b) 
        s = Array.new(0,a.height)
        a.each_row { |i,row| s[i] = row.max }
        a.width.times do |j|
            gauss_partial_pivot_iterate!(a,b,s,j)
        end
        back_substitute_solve!(a,x,b)
    end

    # Gauss Elimination iteration
    # Perform a single iteration, finding a new pivot row using the s-vector, swapping rows, and performing an elimination
    def NumMethods::gauss_partial_pivot_iterate!(a,b,s,j)
            pivot = j
            j.upto(a.height-1) { |i| pivot = i if a[i,j]/s[i] > a[pivot,j]/s[pivot] }
            a.data[j], a.data[pivot] = a.data[pivot], a.data[j]
            b.data[j], b.data[pivot] = b.data[pivot], b.data[j]
            s[j], s[pivot] = s[pivot], s[j]
            gauss_eliminate_iterate!(a,b,j)
    end

    # Perform Gauss Elimination
    # Use the pivot row to eliminate a column of indices below the pivot
    def NumMethods::gauss_eliminate_iterate!(a,b,pivot)
        (pivot+1).upto(a.height-1) do |i|
            ratio = a[i,pivot]/a[pivot,pivot]
            (pivot).upto(a.width-1) do |j|
                a[i,j] -= a[pivot,j]*ratio
            end
            b[i,0] -= b[pivot]*ratio
        end
    end

    # Perform back substitution on an upper-triangular matrix to find the solution to the equation
    def NumMethods::back_substitute_solve!(a,x,b)
        (a.height-1).downto(0) do |i|
            x[i,0] = b[i]
            (i+1).upto(a.width-1) do |j|
                x[i,0] -= a[i,j]*x[j]
            end
            x[i,0] /= a[i,i]
        end
        x
    end

    # Perform forward substitution on a lower-triangular matrix to find the solution to the equation
    def NumMethods::forward_substitute_solve!(a,x,b)
        0.upto(a.height-1) do |i|
            x[i,0] = b[i]
            (i-1).downto(0) do |j|
                x[i,0] -= a[i,j]*x[j]
            end
            x[i,0] /= a[i,i]
        end
        x
    end

    # LU Decomposition solver
    def NumMethods::lu_decomposition_solve(a,b)
        x = NumMethods::Matrix.new(0,b.size)
        NumMethods::valid_equation?(a,x,b) 
        l,u=lu_decompose!(a)
        y = Matrix.new(0,[a.height,1])
        forward_substitute_solve!(l,y,b)
        back_substitute_solve!(u,x,y)
    end

    # Decomposes a into l and u matrices and returns them
    def NumMethods::lu_decompose!(a)
        raise "A must be square to perform LU Decomposition" unless a.square?
        n = a.height - 1
        # Seperate a into l and u
        l = Matrix.new(0.0,a.height)
        u = Matrix.diagonal(1.0,a.height)
        0.upto(n) do |j|
            case j
            when 0:
                0.upto(n) { |i| l[i,0] = a[i,0] }
                1.upto(n) { |i| u[0,i] = a[0,i]/l[0,0] }
            when 1...n:
                j.upto(n) do |i|
                    sum = 0.0
                    0.upto(j-1) { |k| sum += (l[i,k]*u[k,j]) }
                    l[i,j] = a[i,j] - sum;
                end
                (j+1).upto(n) do |k|
                    sum = 0.0 
                    0.upto(j) { |i| sum += (l[j,i]*u[i,k]) }
                    u[j,k] = (a[j,k] - sum) / l[j,j]
                end
            when n:
                l[j,j] = a[j,j]
                0.upto(j-1) { |k| l[j,j] -= (l[j,k]*u[k,j]) }
            end
        end
        return l,u
    end

    # Fits a polynomial to a set of points, iterating through polynomial degrees until the best fit is found
    def NumMethods::least_squares_solve(y,start_degree=0)
        raise "y must be a nx2 matrix" unless y.width == 2
        degree = start_degree
        prev_error = curr_error = Float::MAX
        prev_coeffs = curr_coeffs = nil
        until prev_error < curr_error
            prev_error = curr_error
            prev_coeffs = curr_coeffs
            curr_coeffs = least_squares_fit(y,degree)
            y_approx = Proc.new do |x|
                sum = 0
                curr_coeffs.height.times do |i|
                    sum += curr_coeffs[i]*(x**i)
                end
                sum
            end
            curr_error = least_squares_error(y,y_approx)
            puts "Degree #{degree} error: #{curr_error}"
            degree += 1
        end
        prev_coeffs
    end

    # Fits a polynomial to a set of points in a nx2 matrix [x,y] to a given degree
    # using the Least Squares methods. Returns a column vector of coefficients in increasing order of x
    def NumMethods::least_squares_fit(y,degree)
        m = Matrix::new(0,degree+1)
        b = Matrix::new(0,[degree+1,1])

        # Create the m coefficient matrix using the point data
        0.upto(degree+degree) do |i|
            xsum = 0
            y.each_row { |j,row| xsum += row[0]**i }
            p = if i < degree then i else degree end
            q = if i - degree < 0 then 0 else i - degree end
            until p < 0 or q > degree
                m[p,q] = xsum
                p -= 1
                q += 1
            end
        end


        # Create the b coefficient vector using the point data
        (degree+1).times { |i| b[i,0] = y.data.inject(0) { |sum,row| sum += (row[0]**i)*row[1] } }
        
        # Solve for the coefficients
        lu_decomposition_solve(m,b)
    end

    # Calculates the square error between a set of points [x,y] and an approximating function (Proc object)
    def NumMethods::least_squares_error(y_true,y_approx)
        y_true.data.inject(0) { |sum,row| sum += ( row[1] - y_approx.call(row[0]) )**2 }
    end

    # Calculates the 0 to nth derivatives of a function using backwards difference, returns the solution array
    def NumMethods::back_difference_derivative(func,x0,h,n)
        raise "n must be non-zero to have any meaning"  unless n > 0
        soln_matrix = Matrix.new(0,n)
        soln_matrix.height.times do |i|
            soln_matrix[i,0] = (func.call((x0-(n-1-i)*h)*1.0)-func.call((x0-(n-i)*h)*1.0))/(h*1.0)
        end
        1.upto(soln_matrix.width-1) do |j|
            j.upto(soln_matrix.height-1) do |i|
                soln_matrix[i,j] = (soln_matrix[i,j-1]-soln_matrix[i-1,j-1])/(h*1.0)
            end
        end
        soln_matrix
    end

    # Calculates the 0 to nth derivatives of a function using forwards difference, returns the solution array
    def NumMethods::forward_difference_derivative(func,x0,h,n)
        raise "n must be non-zero to have any meaning" unless n > 0
        soln_matrix = Matrix.new(0,n)
        soln_matrix.height.times do |i|
            soln_matrix[i,0] = (func.call((x0+(i+1)*h)*1.0)-func.call((x0+i*h)*1.0))/(h*1.0)
        end
        1.upto(soln_matrix.width-1) do |j|
            0.upto(soln_matrix.height-1-j) do |i|
                soln_matrix[i,j] = (soln_matrix[i+1,j-1]-soln_matrix[i,j-1])/(h*1.0)
            end
        end
        soln_matrix
    end

    # Calculates the 0 to nth derivatives of a function using central difference, returns the solution array
    def NumMethods::central_difference_derivative(func,x0,h,n)
        raise "n must be non-zero to have any meaning" unless n > 0
        soln_matrix = Matrix.new(0,n)
        soln_matrix.height.times do |i|
            soln_matrix[i,0] = (func.call((x0+(i+1)*h)*1.0)-func.call((x0-(i+1)*h)*1.0))/(h*2.0)
        end
        1.upto(soln_matrix.width-1) do |j|
            0.upto(soln_matrix.height-1-j) do |i|
                soln_matrix[i,j] = (soln_matrix[i+1,j-1]-soln_matrix[i,j-1])/(h*2.0)
            end
        end
        soln_matrix
    end

    # Differentiate a matrix of points using :back, :forward or :central differences
    def NumMethods::differentiate_matrix(pts,h,type)
        if type == :back
            deriv = NumMethods::Matrix.new(0,[pts.height-1,2])
            1.upto(pts.height-1) { |i| deriv[i-1,0] = pts[i,0]; deriv[i-1,1] = (pts[i,1]-pts[i-1,1])/h }
            return deriv
        elsif type == :forward
            deriv = NumMethods::Matrix.new(0,[pts.height-1,2])
            0.upto(pts.height-2) { |i| deriv[i,0] = pts[i,0]; deriv[i,1] = (pts[i+1,1]-pts[i,1])/h }
            return deriv
        elsif type == :central
            deriv = NumMethods::Matrix.new(0,[pts.height-2,2])
            1.upto(pts.height-2) { |i| deriv[i-1,0] = pts[i,0]; deriv[i-1,1] = (pts[i+1,1]-pts[i-1,1])/(2*h) }
            return deriv
        else
            raise "Unknown derivative type"
        end
    end

    # Evaluate a function into a matrix of points over a given domain
    def NumMethods::function_to_matrix(func,a,b,h)
        pts = Matrix.new(0,[((b-a)/h).to_i+1,2])
        pts.each_row { |i,row| row[0] = a+(i*h); row[1] = func.call(row[0]) }
        pts
    end

    # Uses Richardson's Extrapolation to calculate the first derivative of a function to a given order
    # of accuracy, returns the solution array
    def NumMethods::richardsons_derivative(func,x0,h,order)
        raise "order must be non-zero to have any meaning" unless order > 0
        soln_matrix = Matrix.new(0,order)
        soln_matrix.height.times do |i|
            soln_matrix[i,0] = central_difference_derivative(func,x0,h/(2.0**i),1)[0,0]
        end
        1.upto(soln_matrix.width-1) do |j|
            a = (4.0**j)/((4.0**j)-1)
            b = 1.0/((4.0**j)-1)
            0.upto(soln_matrix.height-1-j) do |i|
                soln_matrix[i,j] = soln_matrix[i,j-1]*a-soln_matrix[i+1,j-1]*b
            end
        end
        soln_matrix
    end

    # Calculates a definite integral using the Trap method (from a to b using step size h)
    def NumMethods::trap_integral(func,a,b,h)
        n = ((b-a)/h).abs.to_i
        sum = 0
        sum += func.call(a)
        1.upto(n-1) { |k| sum += 2*func.call(a+k*h) }
        sum += func.call(b)
        sum * (h/2.0)
    end

    # Calculate a definite integral using Romberg integration to a given order (order = cols*2)
    def NumMethods::romberg_integral(func,a,b,cols)
        raise "must calculate a solution matrix of at least 1 column" unless cols > 0
        soln_matrix = Matrix.new(0,cols)
        soln_matrix.height.times do |i|
            soln_matrix[i,0] = trap_integral(func,a,b,(b-a)/(2.0**i))
        end
        1.upto(soln_matrix.width-1) do |j|
            a = (4.0**j)/((4.0**j)-1)
            b = 1.0/((4.0**j)-1)
            j.upto(soln_matrix.height-1) do |i|
                soln_matrix[i,j] = soln_matrix[i,j-1]*a-soln_matrix[i-1,j-1]*b
            end
        end
        soln_matrix[soln_matrix.width-1,soln_matrix.width-1]
    end

    # Finds a solution (matrix of points) to a given derivative dy(x,y) using Heun's Method over the domain [a,b]
    def NumMethods::heuns_method_solve(dy,a,b,y0,h,rel_error=0.0,iterations=1000000)
        soln = Matrix.new(0.0, [(((b-a)/h)+1).ceil,2])
        soln[0,0] = a
        soln[0,1] = y0
        1.upto(soln.height-1) do |i|
            soln[i,0] = a+(i*h)
            begin
                soln[i,1] = heuns_method_iterate(dy,soln[i-1,0],soln[i-1,1],h,rel_error,iterations)
            rescue
                puts "Error in Heun's Method evaluation"
                return soln.truncate([i-1,2])
            end
        end
        soln 
    end

    # Uses Heun's estimation method repeatedly to find y(x0+h) given its derivative
    def NumMethods::heuns_method_iterate(dy,x0,y0,h,rel_error=0.0,iterations=1000000)
        curr_error = Float::MAX 
        prev_error = Float::MAX
        curr_iter = 0
        # Start using Euler's method
        y1_prev = eulers_estimate(dy,x0,y0,h)
        puts "Euler's estimate: #{y1_prev}"
        y1_next = nil
        # Iterate until the estimate is good enough
        until curr_error <= rel_error or curr_iter >= iterations
            y1_next = heuns_estimate(dy,x0,y0,h,y1_prev)
            curr_error = ((y1_next - y1_prev)/y1_next).abs
            y1_prev = y1_next
            curr_iter += 1
            prev_error = curr_error
            if curr_error > prev_error or y1_next.nan? or y1_next.infinite?
                raise "Heun's Method diverging, halting calculation"
            end
        end
        y1_next
    end

    # Refine a previous estimation of a point using Heun's method
    def NumMethods::heuns_estimate(dy,x0,y0,h,y_prev)
        y0 + (h/2)*(dy.call(x0,y0)+dy.call(x0+h,y_prev))
    end

    # Estimates the value of y(x+h) given dy(x,y) using Euler's Method
    def NumMethods::eulers_estimate(dy,x0,y0,h)
        puts "x0:#{x0},y0:#{y0},h:#{h}, estimate:#{y0+h*(dy.call(x0,y0))}"
        y0 + h*(dy.call(x0,y0))
    end

    # Find a solution (matrix of points) to a given derivative dy(x,y) using a Runge-Kutta formula of a given order
    def NumMethods::runge_kutta_solve(dy,a,b,y0,h,order,formula=nil)
        soln = Matrix.new(0.0, [(((b-a)/h)+1).ceil,2])
        soln[0,0] = a
        soln[0,1] = y0
        formula = runge_kutta_formula(order) if formula.nil?
        1.upto(soln.height-1) do |i|
            soln[i,0] = a+(i*h)
            soln[i,1] = formula.call(dy,soln[i-1,0],soln[i-1,1],h,formula)
            # Detect error
            if soln[i,1].nan? or soln[i,1].infinite?
                puts "Error in Runge-Kutta solution!"
                return soln.truncate([i+5,2])
            end
        end
        soln
    end

    # Generates a Runge-Kutta formula to the given order.
    # Any proc objec that returns y1 given |dy,x0,y0,h| can be a Runge-Kutta formula, this method generates
    # the standard ones using the generic form y1 = y0 + h*(a0k0+a1k1+...ankn)
    def NumMethods::runge_kutta_formula(order)
        k = [] # Proc objects (function evaluations)  
        a = [] # coefficients (constants)
        case order
        when 1 # Euler's Method
            a[0] = 1.0
            k[0] = Proc.new { |dy,x0,y0,h| dy.call(x0,y0) }
        when 2
            a[0] = a[1] = 0.5
            k[0] = Proc.new { |dy,x0,y0,h| dy.call(x0,y0) }
            k[1] = Proc.new { |dy,x0,y0,h,k0| dy.call(x0+h, y0 + k0*h ) }
        when 3
            a[0] = a[2] = (1.0/6.0)
            a[1] = (4.0/6.0)
            k[0] = Proc.new { |dy,x0,y0,h| dy.call(x0,y0) }
            k[1] = Proc.new { |dy,x0,y0,h,k0| dy.call(x0+(h/2), y0+k0*(h/2)) }
            k[2] = Proc.new { |dy,x0,y0,h,k0,k1| dy.call(x0+h, y0 - h*k0 + k1*2*h ) }
        when 4
            a[0] = a[3] = (1.0/6.0)
            a[1] = a[2] = (2.0/6.0)
            k[0] = Proc.new { |dy,x0,y0,h| dy.call(x0,y0) }
            k[1] = Proc.new { |dy,x0,y0,h,k0| dy.call(x0+(h/2), y0 + k0*(h/2) ) }
            k[2] = Proc.new { |dy,x0,y0,h,k0,k1| dy.call(x0+(h/2), y0 + k1*(h/2) ) }
            k[3] = Proc.new { |dy,x0,y0,h,k0,k1,k2| dy.call(x0+h, y0 + k2*h) }
        else
            raise "Do not know the order #{order} Runge-Kutta formula"
        end
        # Generate the actual formula now
        Proc.new do |dy,x0,y0,h|
            y1 = y0
            k_vals = []
            order.times do |i|
                k_vals[i] = k[i].call(dy,x0,y0,h,*k_vals)
                y1 += h*a[i]*k_vals[i]
            end
            y1
        end 
    end

    def NumMethods::shooting_method_solve(d2y,a,b,ya,yb,dya1,dya2,h,order,error,formula=nil)
        # We are trying to get a solved solution so that y(x = b) = yb, or in other words, y(x=b) - yb = 0
        # Use the secant method to solve, by wrapping up a proc object that can find a solution given a guess!
        yb_guess = Proc.new do |dya_guess|
            y = shooting_method_iterate(d2y,a,b,ya,yb,dya_guess,h,order,formula)
            puts "dya of #{dya_guess} found yb to be #{y[y.height-1,1]}"
            y[y.height-1,1] - yb # Return how close we are to "zero"
        end
        dya = secant_solve(yb_guess,dya1,dya2,error)
        y = shooting_method_iterate(d2y,a,b,ya,yb,dya,h,order,formula) # Yeah, this is an extra calculation...
        y
    end

    # Uses the shooting method to solve a second-order ODE boundary value problem as a system of two linear ODEs.
    # Requires a Proc object d2y=f(x,dy,y) which should return d2y given x, dy as parameters and the global variable $SHOOTING_Y
    # for y. Yeah, its a hack, but it works with the simple Runge-Kutta solver I already made. This solution goes along x and solves
    # for dy(x=a+ih) using the d2y object, then uses the found solution of dy(x=a+ih) to solve for y. Requires an initial guess for dya
    # and the additional layer of logic to refine the inital guess based on the solution returned is in shooting_method_solve
    def NumMethods::shooting_method_iterate(d2y,a,b,ya,yb,dya,h,order,formula=nil)
        y_soln = Matrix.new(0.0, [(((b-a)/h)+1).ceil,2])
        y_soln[0,0] = a
        y_soln[0,1] = ya
        dy_soln = Matrix.new(0.0, [(((b-a)/h)+1).ceil,2])
        dy_soln[0,0] = a
        dy_soln[0,1] = dya
        dy_proc = Proc.new { |x,y| i = (x-a)/h; dy[i,1] } # returns the value of dy corresponding to x in the solution matrix
        formula = runge_kutta_formula(order) if formula.nil?
        1.upto(y_soln.height-1) do |i|
            $SHOOTING_Y = y_soln[i-1,1]
            dy_soln[i,0] = y_soln[i,0] = a+(i*h)
            dy_soln[i,1] = formula.call(d2y,dy_soln[i-1,0],dy_soln[i-1,1],h,formula) # Solve for the next step of dy/dx
            raise "Error in Runge-Kutta solution" if dy_soln[i,1].nan? or dy_soln[i,1].infinite?
            dy_proc = Proc.new { |x,y| j = (x-a)/h; dy_soln[j,1] } # returns the value of dy corresponding to x in the solution matrix
            y_soln[i,1] = formula.call(dy_proc,y_soln[i-1,0],y_soln[i-1,1],h,formula) # Solve for the next step of y
            raise "Error in Runge-Kutta solution" if dy_soln[i,1].nan? or dy_soln[i,1].infinite?
        end
        y_soln
    end

    # Finds the root of the equation y given two initial x values. Stops after the function at the "root" is less than the
    # specified error
    def NumMethods::secant_solve(y,x1,x2,error)
         x_prev = x1
        y_prev = y.call(x1)
        x_curr = x2
        y_curr = y.call(x2)
        x_next = nil
        curr_error = (error+1)*2
        until curr_error <= error
            x_next = x_curr - (y_curr*(x_curr-x_prev))/(y_curr-y_prev)
            x_prev = x_curr
            y_prev = y_curr
            x_curr = x_next
            y_curr = y.call(x_next)
            curr_error = (y_curr).abs
        end
        x_curr
    end

    # Finds a solution (matrix of points) to a given derivative dy(x,y) using Adams-Bashforth method
    def NumMethods::adams_bashforth_solve(dy,a,b,y0,h,order,formula=nil)
        soln = Matrix.new(0.0, [(((b-a)/h)+1).ceil,2])
        # Solve for [order] initial points using a Runge-Kutta order [order+1] formula, add their values in
        initial_soln = runge_kutta_solve(dy,a,a+(order-1)*h,y0,h,order+1)
        1.upto(order-1) { |i| soln[i,0] = initial_soln[i,0]; soln[i,1] = initial_soln[i,1] }
        formula = adams_bashforth_formula(order) if formula.nil?

        # Iterate through the solution matrix, solving for new points
        order.upto(soln.height-1) do |i|
            soln[i,0] = a+(i*h)
            soln[i,1] = formula.call(dy,order,soln,i-1,h)
            # Detect error
            if soln[i,1].nan? or soln[i,1].infinite?
                puts "Error in Adams-Bashforth solution!"
                return soln.truncate([i-1,2])
            end
        end
        soln
    end

    # Generates a Adams-Bashforth formula for the given order
    # Any Proc that returns y1 given dy, an order, calculated values of at least [order] points, the current row
    # and the step size can be an Adams-Bashforth formula. This method generates a number of known ones
    def NumMethods::adams_bashforth_formula(order)
        b = []
        case order
        when 1 
            b[0] = 1.0 # Euler's Method
        when 2 # O(h**3)
            b[0] = (3.0/2.0)
            b[1] = (-1.0/2.0)
        when 3 # O(h**4)
            b[0] = (23.0/12.0)
            b[1] = (-16.0/12.0)
            b[2] = (5.0/12.0)
        else
            raise "Do not know the order #{order} Adams-Bashforth formula"
        end
        # Generate the formula
        Proc.new do |dy,order,soln,i,h|
            y1 = soln[i,1]
            0.upto(order-1) { |k| y1 += h*b[k]*dy.call(soln[i-k,0],soln[i-k,1]) }
            y1
        end
    end

    # Finds a solution to an ODE using Adams-Moulton's method (with Adams-Bashforth as the predictor). Continues
    # refining each step until a specific accuracy is retrieved. Assumes the Bashforth & Moulton orders to be identical
    def NumMethods::adams_moulton_solve(dy,a,b,y0,h,order,bashforth=nil,moulton=nil,rel_error=0.0,iterations=1000000)
        soln = Matrix.new(0.0, [(((b-a)/h)+1).ceil,2])
        # Solve for [order] initial points using a Runge-Kutta order [order+1] formula, add their values in
        initial_soln = runge_kutta_solve(dy,a,a+(order-1)*h,y0,h,order+1)
        0.upto(order-1) { |i| soln[i,0] = initial_soln[i,0]; soln[i,1] = initial_soln[i,1] }
        bashforth = adams_bashforth_formula(order) if bashforth.nil?
        moulton = adams_moulton_formula(order) if moulton.nil?

        # Iterate through the solution matrix, solving for new points
        order.upto(soln.height-1) do |i|
            soln[i,0] = a+(i*h)
            begin
                soln[i,1] = adams_moulton_iterate(dy,order,soln,i-1,h,bashforth,moulton,rel_error,iterations)
            rescue
                puts "Error in Adams-Moulton solution!"
                return soln.truncate([i-1,2])
            end
        end
        soln
    end

    # Performs multiple iterations of the Adams-Moulton method to find the next point given the previous solutions
    # Requires two formulas: the bashforth predictor and the moulton corrector
    def NumMethods::adams_moulton_iterate(dy,order,soln,i,h,bashforth=nil,moulton=nil,rel_error=0.0,iterations=1000000)
        bashforth = adams_bashforth_formula(order) if bashforth.nil?
        moulton = adams_moulton_formula(order) if moulton.nil?
        curr_error = Float::MAX 
        prev_error = Float::MAX 
        curr_iter = 0
        y1_prev = bashforth.call(dy,order,soln,i,h)
        y1_next = nil
        # Iterate until the estimate is good enough
        until curr_error <= rel_error or curr_iter >= iterations
            y1_next = moulton.call(dy,order,soln,i,h,y1_prev)
            curr_error = ((y1_next - y1_prev)/y1_next).abs
            y1_prev = y1_next
            if curr_error > prev_error or y1_next.nan? or y1_next.infinite?
                raise "Moulton's Method diverging, halting calculation"
            end
            prev_error = curr_error
            curr_iter += 1
        end
        y1_next
    end

    # Generates an Adams-Moulton formula of a given order. An Adams-Moulton formula is any Proc that returns a better
    # value for y1 given dy, its order, the previously calculated values, and the current row of the solution and a previous
    # estimate of y_next. This method generates some known formulas
    def NumMethods::adams_moulton_formula(order)
        b = []
        case order
        when 1 # Backwards Euler's
            b[0] = 1.0
        when 2
            b[0] = b[1] = (1.0/2.0)
        when 3
            b[0] = (5.0/12.0)
            b[1] = (8.0/12.0)
            b[2] = (-1.0/12.0)
        else
            raise "Do not know the order #{order} Adams-Moulton formula"
        end
        # Generate the formula
        Proc.new do |dy,order,soln,i,h,y1_estimate|
            y1 = soln[i,1]
            y1 += h*b[0]*dy.call(soln[i,0]+h,y1_estimate)
            1.upto(order-1) { |k| y1 += h*b[k]*dy.call(soln[i+1-k,0],soln[i+1-k,1]) }
            y1
        end
    end
end

